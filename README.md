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
pipeline, prompts and command vocabulary can be settled before any of it is
wired to global hotkeys.

**One control.** Hold `Ctrl+Space` (or the button) and talk. What happens
depends on the selection, which is how you already think about it:

| | |
|---|---|
| nothing selected | speech is **dictated** at the cursor |
| text selected | speech is an **instruction about that text** |

Hold to talk, release to stop. A tap shorter than 350 ms latches instead, so you
can toggle for a long dictation rather than holding the key down.

### Voice commands

Phrases handled locally — no LLM call, no latency, no chance of being
reinterpreted. `GET /api/commands` serves the list; clients match it themselves,
because the same word means different things in a browser and on a desktop.

`never mind` · `scratch that` · `undo` · `delete that` · `all caps` ·
`lowercase that` · `title case` · `new paragraph` …

**Saying "never mind" cancels** — including when it trails a real instruction
("make it bold, no, scratch that"). `Esc` does the same from the keyboard, at
every stage: recording, transcribing, during the grace period, and with the LLM
request in flight.

### Modes

Each mode carries its own edit prompt, dictation prompt, model and temperature:
**Prose**, **Email**, **Commit message**, **Chat/Slack**, **Code comment**. They
live on the server (`GET/PUT /api/modes`) rather than in browser storage, so the
desktop client will inherit whatever you tune here — and will be able to pick a
mode from the focused application.

Dictating "um this fixes the thing where the parser was crashing on empty input"
in commit mode yields `Fix parser crashing on empty input`.

### Dictation cleanup

With **Clean up dictation** on, raw speech goes through the mode's dictation
prompt first. It resolves spoken self-corrections and drops filler, without
touching your register:

> so um I think we should ship it on Tuesday no wait not Tuesday lets do
> Wednesday because the the release window is is better

becomes *"I think we should ship it on Wednesday because the release window is
better."* Turn it off for verbatim dictation.

### Learned vocabulary

When an edit replaces a garbled term with a real one, the term is remembered
along with what the recogniser heard. After two sightings it is promoted into
the hotwords sent with every recording, so the mistake stops recurring. Stored
server-side in `data/store.json`, listed and editable in the left rail.

Fixing "cooper netties" once records `Kubernetes` ← *heard as "cooper netties"*.

### Review before applying

**Pause to show what was heard** (default on) displays the instruction for a
grace period, then applies on its own — a window to bail out of, not a gate that
waits for you.

| | |
|---|---|
| `Enter` | apply now, from wherever focus is |
| `Esc` | cancel, no API call made |
| type in the box | stops the timer, so you can fix a misheard word |

**Retry** undoes the last edit and re-runs the same instruction, for comparing
models or prompt changes. Everything is undoable.

### Keys and storage

The OpenRouter call is proxied through the backend (`POST /api/llm/edit`) rather
than made from the browser, so a desktop client can hit the same endpoint with
the same modes. Set `OPENROUTER_API_KEY` in `.env`, or type a key into the page
(it goes to `localStorage`; the page says which mode it is in). Modes and
vocabulary persist in `./data`, which is bind-mounted into the backend.

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
docker compose up -d --build          # first time only
./run.sh                              # every day after that
```

`run.sh` waits for the checkpoint to load and prints the URLs:

| | |
|---|---|
| `./run.sh` | start, wait for the model, print the URLs |
| `./run.sh stop` | shut down and free the GPU |
| `./run.sh status` | what is running, and whether the model is loaded |
| `./run.sh logs` | follow the backend log |
| `./run.sh restart` | stop, then start |

Stopping keeps everything that matters: the weights in `./models`, your modes
and learned vocabulary in `./data`, and your API key in `.env`. A warm restart
takes about 10 seconds.

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
