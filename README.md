# VibeVoice Streaming ASR — local GPU demo

A two-container demo of [`microsoft/VibeVoice-ASR-Streaming-7B`](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B),
the streaming speech-to-text model that transcribes **who said what** while the
audio is still arriving.

| Container | What it is | Port |
|---|---|---|
| `backend` | FastAPI wrapper around the checkpoint, running on your GPU | `127.0.0.1:8001` |
| `frontend` | nginx serving the demo UI and proxying `/api` to the backend | `127.0.0.1:8080` |

Two pages:

- **http://localhost:8080** — the ASR demo (live mic, files, examples, hotword A/B)
- **http://localhost:8080/editor.html** — **voice editing**: select text, say what to
  change, an LLM rewrites it in place

---

## What you can try

- **Live** — record from the microphone and watch the transcript appear as you
  talk. The model emits text every ~2.9 s of audio with a ~0.5 s lookahead, so
  words land while you are still speaking.
- **File** — drop in any audio or video file. ffmpeg decodes it server-side, so
  mp3, wav, m4a, flac, ogg, opus, mp4, webm and mov all work.
- **Examples** — three bundled clips from the upstream repo: a two-speaker
  conversation, a hotword-heavy clip, and singing.
- **A/B** — transcribe the same audio twice, once cold and once with hotwords,
  with the hotword hits highlighted. This is the clearest way to see what
  `context_info` actually does.

Along the way the right-hand rail shows chunk latency against the real-time
budget, the running real-time factor, token counts and GPU memory. Tick
**Show raw model output** in the left rail to see the untouched decoder output,
speaker prefixes and all.

### Measured on this box (RTX 3090, bf16, sdpa)

| | |
|---|---|
| Weights resident | 17.4 GB of 24 GB |
| Checkpoint load | ~4 s with the weights in page cache; the first load after a reboot is bounded by disk read speed |
| Chunk latency | 280–570 ms against a 2.93 s budget |
| Time to first token | ~65 ms after a chunk closes |
| Real-time factor | **0.13 – 0.20** (5–8× faster than the audio) |

Because RTF is well under 1, live transcription keeps up with the microphone
with room to spare — the `backlog` events stay at zero.

---

## Voice editing

`editor.html` is a browser prototype of a desktop dictation workflow, so the
pipeline and the prompt can be tuned before any of it is wired to global
hotkeys.

- **Dictate** (`Ctrl+Shift+D`) — verbatim speech at the cursor.
- **Voice edit** (`Ctrl+Shift+E`) — select a passage, say what to change. With
  nothing selected it edits the whole document.

The transcribed instruction and the selected text go to OpenRouter through the
backend (`POST /api/llm/edit`) — the browser never calls OpenRouter directly, so
a desktop client can hit the same endpoint later. Set `OPENROUTER_API_KEY` in
`.env`, or type a key into the page (it goes to `localStorage`; the page says
which mode it is in).

**Cancelling.** Speech recognition mishears, so `Esc` means *stop, change
nothing* at every stage — while recording, while transcribing, and while the
LLM call is in flight. With **Review the instruction before applying** on (the
default), what was heard is shown in an editable box first: fix a wrong word and
press Enter, or `Esc` to throw it away without spending an API call. **Retry**
undoes the last edit and re-runs the same instruction, which is how you compare
models or prompt changes. Everything is undoable.

The system prompt is an editable textarea in the left rail — it is the entire
behaviour of the edit pass, so tune it there and re-run.

Words from the document are harvested as ASR hotwords, along with common editing
verbs, so an instruction naming a term already on screen is more likely to come
through intact.

---

## Requirements

- NVIDIA GPU with **≥ 20 GB** free VRAM (the checkpoint is 17.4 GB in bf16;
  verified on an RTX 3090)
- NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (`docker info | grep nvidia` should list the runtime)
- Docker with Compose v2
- ~20 GB of disk for the weights

## Setup

### 1. Download the weights

They are **not** baked into the image — they are bind-mounted, so rebuilding the
app never re-copies 17 GB.

```bash
pip install -U "huggingface_hub[cli]"
hf download microsoft/VibeVoice-ASR-Streaming-7B \
  --local-dir ./models/VibeVoice-ASR-Streaming-7B
```

You should end up with `models/VibeVoice-ASR-Streaming-7B/` containing
`config.json`, `preprocessor_config.json` and eight `model-0000N-of-00008.safetensors`
shards.

### 2. Build and start

```bash
docker compose up -d --build
docker compose logs -f backend        # watch the checkpoint load
```

The first start reads 17 GB off disk and onto the GPU; later restarts take a few
seconds because the weights are still in the page cache. The UI shows a loading
banner until `/api/health` reports `ready`, then unlocks itself.

### 3. Open the UI

<http://localhost:8080>

The microphone needs a secure context; `localhost` counts as one, so no HTTPS
setup is required.

---

## Configuration

Copy `.env.example` to `.env` to override any of these:

| Variable | Default | Meaning |
|---|---|---|
| `WEB_PORT` | `8080` | Host port for the UI |
| `API_PORT` | `8001` | Host port for the backend API |
| `MODEL_DIR` | `./models` | Host directory mounted at `/models` |
| `ATTN_IMPLEMENTATION` | `sdpa` | `sdpa`, `eager`, or `flash_attention_2` if you add flash-attn to the image |
| `MAX_UPLOAD_MB` | `200` | Upload size cap |
| `MAX_AUDIO_SECONDS` | `1800` | Longer audio is truncated |

Both ports bind to `127.0.0.1` only. Drop the prefix in `docker-compose.yml` if
you want the demo reachable from your LAN.

---

## API

The backend is usable on its own at `http://localhost:8001` (change with `API_PORT`).

### `GET /api/health`

```json
{ "status": "ready", "device": "cuda", "dtype": "bfloat16",
  "gpu": { "name": "NVIDIA GeForce RTX 3090", "reserved_gb": 17.1, "total_gb": 25.4 },
  "frames": { "chunk_seconds": 2.9333, "lookahead_seconds": 0.5333, "sample_rate": 24000 } }
```

### `POST /api/transcribe` → Server-Sent Events

Multipart form: `file`, plus optional `context_info`, `temperature`,
`max_new_tokens`, `repetition_penalty`.

```bash
curl -N http://localhost:8001/api/transcribe \
  -F file=@samples/demo1-chat.mp3 \
  -F context_info="VibeVoice, Microsoft"
```

Event stream:

| `type` | Payload |
|---|---|
| `meta` | duration, chunk count, frame config, echoed options |
| `delta` | one token's worth of text, plus the segment and speaker it belongs to |
| `chunk` | one chunk finished: index, audio span, latency, running stats |
| `done` | final text, speaker segments, raw decoder output, stats |
| `error` | message |

`delta` is what makes the UI type live — the backend re-implements the upstream
chunk loop so it can forward each token as it is sampled, rather than one blob
per chunk.

### `POST /api/transcribe/json`

Same form fields, one JSON body with `text`, `segments`, `chunks`, `raw`, `stats`.

### `POST /api/compare`

Form fields `file` and `context_info`. Transcribes twice — without hotwords then
with — tagging every event with `run: "without" | "with"`.

### `WS /api/ws/live`

1. Send one JSON text frame: `{"context_info": null, "temperature": 0, "max_new_tokens": 256}`
2. Send raw **mono float32 little-endian PCM at 24 kHz** as binary frames.
3. Send the text frame `end` to flush the tail.

You get back the same event objects as the SSE stream, plus `backlog` events
telling you how much audio is queued ahead of the GPU.

### Command-line check

```bash
python3 scripts/smoke_test.py                              # health/config/samples
python3 scripts/smoke_test.py --file samples/demo1-chat.mp3
python3 scripts/smoke_test.py --live samples/demo1-chat.mp3   # real-time pacing
```

---

## How the streaming works

The checkpoint's `preprocessor_config.json` fixes the cadence, and the backend
reads it rather than hard-coding anything:

```
speech_tok_compress_ratio 3200 @ 24 kHz   ->  one acoustic frame = 133.3 ms
chunk_frames     22                       ->  2.933 s of audio per step
lookahead_frames  4                       ->  0.533 s of right context
```

Each step encodes `chunk + lookahead` samples, appends them to a KV cache that
persists for the whole session, and decodes text until the model emits
`<|text_chunk_end|>`. The window then advances by `chunk` only — the lookahead
stays in the buffer as the next chunk's leading audio. Because the cache is never
rebuilt, cost per chunk stays flat and the model keeps the full conversation as
context, which is what lets speaker labels stay consistent across a long
recording.

Speaker attribution arrives inline as `<|object_ref_start|>Speaker 0<|object_ref_end|>`.
Upstream's helpers strip those markers; this backend parses them into segments
instead, tolerating markers that straddle a token boundary.

---

## Layout

```
backend/
  app/engine.py       model loading, chunk loop, transcript parsing
  app/main.py         FastAPI routes: SSE, WebSocket, samples, health
  app/audio.py        ffmpeg decode to mono float32 @ 24 kHz
  vendor/vibevoice/   vendored from github.com/microsoft/VibeVoice (MIT)
frontend/
  web/                the single-page UI (no build step)
  nginx.conf          static files + /api proxy with WS upgrade and SSE buffering off
samples/              demo clips from the upstream repo
scripts/smoke_test.py CLI exercise of both streaming paths
models/               the checkpoint (not in git)
```

## Troubleshooting

**`status: loading` forever** — `docker compose logs backend`. A missing
`preprocessor_config.json` means the download is incomplete; re-run `hf download`
(it resumes).

**CUDA out of memory** — the model needs ~17 GB plus a KV cache. Free the GPU
(`nvidia-smi`) or set `DEVICE=cpu` in the compose file to check correctness
slowly.

**`could not select device driver "nvidia"`** — the NVIDIA Container Toolkit is
not installed or Docker was not restarted after installing it.

**Microphone button does nothing** — the browser blocks capture outside a secure
context. Use `http://localhost:8080`, not the machine's LAN IP. If you see
"Microphone blocked", grant the permission for the site in Chrome's address bar
and click Record again.

**Port already allocated** — something else owns `8080` or `8001`. Set `WEB_PORT`
or `API_PORT` in `.env` and `docker compose up -d` again.

## Licences

The demo code here is yours to do as you like with. The model is MIT
(Microsoft), and `backend/vendor/vibevoice/` is vendored unmodified from
[microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) under MIT —
see `backend/vendor/LICENSE.vibevoice`.
