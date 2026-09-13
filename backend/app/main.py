"""FastAPI wrapper around microsoft/VibeVoice-ASR-Streaming-7B.

Three ways in:
  * POST /api/transcribe        -- upload a file, get SSE events as it decodes
  * POST /api/transcribe/json   -- upload a file, get one JSON result
  * WS   /api/ws/live           -- push mic PCM, get events back as you speak
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Optional

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import audio as audio_utils
from . import llm
from .engine import DecodeOptions, VibeVoiceEngine

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("vibevoice.api")

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/VibeVoice-ASR-Streaming-7B")
DEVICE = os.environ.get("DEVICE", "cuda")
ATTN_IMPL = os.environ.get("ATTN_IMPLEMENTATION", "sdpa")
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", "/samples")
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "200"))
MAX_AUDIO_SECONDS = float(os.environ.get("MAX_AUDIO_SECONDS", "1800"))

LANGUAGES = [
    {"code": "en", "name": "English"},
    {"code": "zh", "name": "Chinese"},
    {"code": "es", "name": "Spanish"},
    {"code": "pt", "name": "Portuguese"},
    {"code": "de", "name": "German"},
    {"code": "ja", "name": "Japanese"},
    {"code": "ko", "name": "Korean"},
    {"code": "fr", "name": "French"},
    {"code": "ru", "name": "Russian"},
    {"code": "it", "name": "Italian"},
]

SAMPLES = [
    {
        "id": "demo1-chat",
        "file": "demo1-chat.mp3",
        "title": "Two-speaker conversation",
        "blurb": "Shows speaker-attributed streaming: the transcript splits into speakers as it arrives.",
        "hotwords": "",
    },
    {
        "id": "demo3-hotwords",
        "file": "demo3-hotwords.wav",
        "title": "Hotword biasing",
        "blurb": "Names and jargon the model gets wrong cold. Run it twice — once with the hotwords, once without.",
        "hotwords": "VibeVoice, Microsoft, ASR, Qwen, tokenizer",
    },
    {
        "id": "demo2-song",
        "file": "demo2-song.mp3",
        "title": "Singing",
        "blurb": "Sung vocals over music — a harder case than clean speech.",
        "hotwords": "",
    },
]

engine = VibeVoiceEngine(MODEL_PATH, device=DEVICE, attn_implementation=ATTN_IMPL)
_load_error: Optional[str] = None
_load_started = time.time()

@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_load_model, daemon=True, name="model-loader").start()
    yield


app = FastAPI(title="VibeVoice Streaming ASR API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_model() -> None:
    global _load_error
    try:
        log.info("loading checkpoint from %s (device=%s, attn=%s)", MODEL_PATH, DEVICE, ATTN_IMPL)
        engine.load()
        log.info("model ready in %.1fs", engine.load_seconds or 0.0)
    except Exception as exc:  # noqa: BLE001 - surfaced through /api/health
        _load_error = f"{type(exc).__name__}: {exc}"
        log.exception("model failed to load")




def _require_ready() -> None:
    if _load_error:
        raise HTTPException(status_code=503, detail=f"model failed to load: {_load_error}")
    if not engine.ready:
        raise HTTPException(status_code=503, detail="model is still loading")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> JSONResponse:
    if _load_error:
        status = "error"
    elif engine.ready:
        status = "ready"
    else:
        status = "loading"
    payload = engine.info()
    payload.update(
        status=status,
        error=_load_error,
        uptime_seconds=round(time.time() - _load_started, 1),
    )
    return JSONResponse(payload, status_code=200 if status != "error" else 503)


@app.get("/api/config")
async def config() -> JSONResponse:
    _require_ready()
    return JSONResponse(
        {
            "model": "microsoft/VibeVoice-ASR-Streaming-7B",
            "frames": engine.frames.as_dict(),
            "languages": LANGUAGES,
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_audio_seconds": MAX_AUDIO_SECONDS,
            "defaults": {"temperature": 0.0, "max_new_tokens": 256, "repetition_penalty": 1.0},
        }
    )


@app.get("/api/samples")
async def samples() -> JSONResponse:
    out = []
    for sample in SAMPLES:
        path = os.path.join(SAMPLES_DIR, sample["file"])
        if not os.path.isfile(path):
            continue
        out.append({**sample, "url": f"/api/samples/{sample['id']}/audio", "bytes": os.path.getsize(path)})
    return JSONResponse(out)


@app.get("/api/samples/{sample_id}/audio")
async def sample_audio(sample_id: str):
    for sample in SAMPLES:
        if sample["id"] == sample_id:
            path = os.path.join(SAMPLES_DIR, sample["file"])
            if not os.path.isfile(path):
                raise HTTPException(status_code=404, detail="sample file is missing")
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="unknown sample")


# ---------------------------------------------------------------------------
# Voice editing (OpenRouter)
# ---------------------------------------------------------------------------


@app.get("/api/llm/status")
async def llm_status() -> JSONResponse:
    """Whether the server holds a key, so the page can skip asking for one."""
    return JSONResponse(
        {
            "server_key": llm.server_key() is not None,
            "default_model": llm.DEFAULT_MODEL,
            "default_system_prompt": llm.DEFAULT_SYSTEM_PROMPT,
            "suggested": llm.SUGGESTED_MODELS,
        }
    )


@app.get("/api/llm/models")
async def llm_models(refresh: bool = False) -> JSONResponse:
    try:
        models = await asyncio.to_thread(llm.list_models, refresh)
    except llm.LLMError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    return JSONResponse(models)


@app.post("/api/llm/edit")
async def llm_edit(payload: dict = Body(...)) -> JSONResponse:
    """Apply a spoken instruction to a passage of text."""
    text = payload.get("text") or ""
    instruction = payload.get("instruction") or ""
    try:
        result = await asyncio.to_thread(
            llm.edit_text,
            text,
            instruction,
            model=payload.get("model"),
            system_prompt=payload.get("system_prompt"),
            temperature=float(payload.get("temperature") or 0.2),
            api_key=payload.get("api_key"),
        )
    except llm.LLMError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Bridging blocking inference to async streaming
# ---------------------------------------------------------------------------


_DONE = object()


async def _stream_events(work: Callable[[Callable[[dict], None]], None]) -> AsyncIterator[dict]:
    """Run `work` on a thread and yield the events it pushes."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def runner() -> None:
        try:
            work(on_event)
        except Exception as exc:  # noqa: BLE001 - forwarded to the client
            log.exception("inference failed")
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    threading.Thread(target=runner, daemon=True, name="inference").start()
    while True:
        event = await queue.get()
        if event is _DONE:
            return
        yield event


async def _read_upload(file: UploadFile) -> bytes:
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    data = await file.read()
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"file is larger than {MAX_UPLOAD_MB:.0f} MB")
    return data


def _prepare(data: bytes) -> tuple[np.ndarray, bool]:
    try:
        audio = audio_utils.decode_bytes(data, target_sr=engine.frames.sample_rate)
    except audio_utils.AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return audio_utils.trim(audio, engine.frames.sample_rate, MAX_AUDIO_SECONDS)


# ---------------------------------------------------------------------------
# File transcription
# ---------------------------------------------------------------------------


@app.post("/api/transcribe")
async def transcribe_sse(
    file: UploadFile = File(...),
    context_info: str = Form(""),
    temperature: float = Form(0.0),
    max_new_tokens: int = Form(256),
    repetition_penalty: float = Form(1.0),
):
    """Stream the transcript back as Server-Sent Events while it is produced."""
    _require_ready()
    data = await _read_upload(file)
    audio, truncated = _prepare(data)
    options = DecodeOptions.from_dict(
        {
            "context_info": context_info,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
    )

    def work(on_event: Callable[[dict], None]) -> None:
        if truncated:
            on_event({"type": "notice", "message": f"audio truncated to {MAX_AUDIO_SECONDS:.0f}s"})
        engine.transcribe_array(audio, options, on_event)

    async def body() -> AsyncIterator[bytes]:
        yield b": open\n\n"
        async for event in _stream_events(work):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/api/transcribe/json")
async def transcribe_json(
    file: UploadFile = File(...),
    context_info: str = Form(""),
    temperature: float = Form(0.0),
    max_new_tokens: int = Form(256),
    repetition_penalty: float = Form(1.0),
) -> JSONResponse:
    """Blocking variant -- one JSON body with the finished transcript."""
    _require_ready()
    data = await _read_upload(file)
    audio, truncated = _prepare(data)
    options = DecodeOptions.from_dict(
        {
            "context_info": context_info,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
    )
    chunks: list[dict] = []
    result: dict = {}

    def work(on_event: Callable[[dict], None]) -> None:
        def collect(event: dict) -> None:
            if event["type"] == "chunk":
                chunks.append(event)
            elif event["type"] == "done":
                result.update(event)
        engine.transcribe_array(audio, options, collect)

    failure: Optional[dict] = None
    async for event in _stream_events(work):
        if event.get("type") == "error":
            failure = event
    if failure is not None:
        raise HTTPException(status_code=500, detail=failure.get("message", "inference failed"))
    if not result:
        raise HTTPException(status_code=500, detail="inference produced no result")
    result["chunks"] = chunks
    result["truncated"] = truncated
    return JSONResponse(result)


@app.post("/api/compare")
async def compare_hotwords(
    file: UploadFile = File(...),
    context_info: str = Form(...),
    temperature: float = Form(0.0),
    max_new_tokens: int = Form(256),
):
    """Transcribe the same audio twice -- without hotwords, then with them."""
    _require_ready()
    data = await _read_upload(file)
    audio, _ = _prepare(data)

    def work(on_event: Callable[[dict], None]) -> None:
        for label, ctx in (("without", None), ("with", context_info)):
            on_event({"type": "run_start", "run": label, "context_info": ctx})
            options = DecodeOptions.from_dict(
                {"context_info": ctx, "temperature": temperature, "max_new_tokens": max_new_tokens}
            )

            def tagged(event: dict, _label=label) -> None:
                event = dict(event)
                event["run"] = _label
                if event["type"] == "done":
                    event["type"] = "run_done"
                on_event(event)

            engine.transcribe_array(audio, options, tagged)
        on_event({"type": "done"})

    async def body() -> AsyncIterator[bytes]:
        yield b": open\n\n"
        async for event in _stream_events(work):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Live microphone
# ---------------------------------------------------------------------------


@app.websocket("/api/ws/live")
async def ws_live(ws: WebSocket) -> None:
    await ws.accept()
    if not engine.ready:
        await ws.send_json({"type": "error", "message": _load_error or "model is still loading"})
        await ws.close()
        return

    try:
        options = DecodeOptions.from_dict(json.loads(await ws.receive_text()))
    except Exception:
        await ws.send_json({"type": "error", "message": "expected a JSON options frame first"})
        await ws.close()
        return

    frames = engine.frames
    loop = asyncio.get_running_loop()
    # Building the initial KV cache touches the GPU and takes the engine lock,
    # so it must not run on the event loop thread.
    try:
        session = await asyncio.to_thread(engine.new_session, options)
    except Exception as exc:  # noqa: BLE001
        await ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        await ws.close()
        return
    outbox: asyncio.Queue = asyncio.Queue()
    buffer = np.zeros(0, dtype=np.float32)
    buffer_lock = threading.Lock()
    finished = asyncio.Event()
    flushing = False

    await ws.send_json(
        {
            "type": "meta",
            "frames": frames.as_dict(),
            "options": {
                "context_info": options.context_info,
                "temperature": options.temperature,
                "max_new_tokens": options.max_new_tokens,
            },
        }
    )

    def on_event(event: dict) -> None:
        loop.call_soon_threadsafe(outbox.put_nowait, event)

    def take_window() -> Optional[tuple[np.ndarray, int]]:
        """Pop one window worth of audio, or pad the tail when flushing."""
        nonlocal buffer
        with buffer_lock:
            available = len(buffer)
            if available >= frames.window_samples:
                window = buffer[: frames.window_samples].copy()
                advance = frames.chunk_samples
            elif flushing and available > 0:
                window = np.zeros(frames.window_samples, dtype=np.float32)
                window[:available] = buffer
                advance = min(available, frames.chunk_samples)
            else:
                return None
            # Keep the lookahead: it is the next chunk's leading audio.
            buffer = buffer[frames.chunk_samples :]
            return window, advance

    def backlog_seconds() -> float:
        with buffer_lock:
            return max(0.0, (len(buffer) - frames.lookahead_samples) / frames.sample_rate)

    async def worker() -> None:
        try:
            while True:
                item = take_window()
                if item is None:
                    if flushing:
                        session.finalize(on_event)
                        return
                    await asyncio.sleep(0.02)
                    continue
                window, advance = item
                await asyncio.to_thread(session.process_window, window, advance, on_event)
                on_event({"type": "backlog", "seconds": round(backlog_seconds(), 2)})
        except Exception as exc:  # noqa: BLE001
            log.exception("live session failed")
            on_event({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            loop.call_soon_threadsafe(finished.set)

    async def sender() -> None:
        while True:
            event = await outbox.get()
            await ws.send_text(json.dumps(event, ensure_ascii=False))
            if event.get("type") in ("done", "error"):
                return

    worker_task = asyncio.create_task(worker())
    sender_task = asyncio.create_task(sender())

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            payload = message.get("bytes")
            if payload is not None:
                pcm = np.frombuffer(payload, dtype="<f4")
                with buffer_lock:
                    buffer = np.concatenate([buffer, pcm])
                continue
            text = message.get("text")
            if text == "end":
                flushing = True
                await asyncio.wait_for(finished.wait(), timeout=600)
                await asyncio.wait_for(sender_task, timeout=30)
                break
            if text == "ping":
                await ws.send_json({"type": "pong", "backlog": round(backlog_seconds(), 2)})
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log.warning("live websocket ended: %s", exc)
    finally:
        flushing = True
        worker_task.cancel()
        sender_task.cancel()
        session.release()
        try:
            await ws.close()
        except Exception:
            pass
