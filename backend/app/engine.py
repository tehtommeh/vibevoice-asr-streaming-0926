"""Inference engine around microsoft/VibeVoice-ASR-Streaming-7B.

The upstream `streaming_generate*` helpers decode each chunk with
`skip_special_tokens=True`, which throws away the speaker markers the model
emits. We re-implement the same chunk loop here so we keep the raw ids: that
buys speaker attribution *and* token-level streaming to the browser instead of
one blob per chunk.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np
import torch

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

# Markers the model wraps the speaker label in. Upstream strips these; we parse
# them instead.
SPEAKER_OPEN = "<|object_ref_start|>"
SPEAKER_CLOSE = "<|object_ref_end|>"

_STRIP_TOKENS = (
    "<|text_chunk_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|speech_start|>",
    "<|speech_end|>",
    "<|speech_pad|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
)

def clean_text(text: str) -> str:
    for tok in _STRIP_TOKENS:
        text = text.replace(tok, "")
    return text


@dataclass
class Segment:
    """One contiguous run of speech attributed to a single speaker."""

    speaker: Optional[str]
    text: str = ""
    chunk_start: int = 0
    chunk_end: int = 0

    def as_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "text": self.text.strip(),
            "chunk_start": self.chunk_start,
            "chunk_end": self.chunk_end,
        }


# This checkpoint labels turns with a plain "\n Speaker 0:" prefix. It arrives a
# token at a time, so "Speaker", " 0" and ":" routinely land in three separate
# pieces -- the scanner below holds back any tail that could still become one.
SPEAKER_WORD = "Speaker"
_SPEAKER_FULL = re.compile(r"Speaker[ \t]*[0-9A-Za-z_\-]{0,24}[ \t]*:")
_SPEAKER_PARTIAL = re.compile(r"Speaker[ \t]*[0-9A-Za-z_\-]{0,24}[ \t]*\Z")


def _partial_suffix_len(buf: str, marker: str) -> int:
    """Length of the longest suffix of `buf` that is a proper prefix of `marker`."""
    for n in range(min(len(marker) - 1, len(buf)), 0, -1):
        if buf.endswith(marker[:n]):
            return n
    return 0


class TranscriptBuilder:
    """Turns the raw token stream into speaker-attributed segments.

    Fed one decoded piece at a time. Returns (segment_index, appended_text) so
    the browser can append in place rather than re-render the transcript.
    """

    def __init__(self) -> None:
        self.segments: list[Segment] = []
        self.raw = ""
        self._buf = ""
        self._last_char: Optional[str] = None
        self._chunk = 0

    def set_chunk(self, index: int) -> None:
        self._chunk = index

    @property
    def text(self) -> str:
        return "\n".join(
            (f"{s.speaker}: {s.text.strip()}" if s.speaker else s.text.strip())
            for s in self.segments
            if s.text.strip()
        )

    @property
    def plain_text(self) -> str:
        """Just the words, no speaker labels -- what dictation wants."""
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    # -- scanning ---------------------------------------------------------
    def feed(self, piece: str) -> list[tuple[int, str]]:
        self.raw += piece
        self._buf += piece
        return self._scan(final=False)

    def flush(self) -> list[tuple[int, str]]:
        out = self._scan(final=True)
        return out

    def _find_speaker_word(self, buf: str) -> int:
        """First 'Speaker' sitting at a word boundary, or -1."""
        start = 0
        while True:
            index = buf.find(SPEAKER_WORD, start)
            if index == -1:
                return -1
            prev = buf[index - 1] if index > 0 else self._last_char
            if prev is None or prev.isspace():
                return index
            start = index + 1

    def _boundary_partial_len(self, buf: str) -> int:
        """Held-back tail that could still grow into a speaker prefix."""
        # A complete-but-colonless "Speaker 0" at the tail.
        index = self._find_speaker_word(buf)
        if index != -1 and _SPEAKER_PARTIAL.match(buf, index):
            return len(buf) - index
        # A partial word ("Spea") at the tail, at a valid boundary.
        keep = _partial_suffix_len(buf, SPEAKER_WORD)
        while keep:
            index = len(buf) - keep
            prev = buf[index - 1] if index > 0 else self._last_char
            if prev is None or prev.isspace():
                return keep
            keep = _partial_suffix_len(buf[: len(buf) - 1], SPEAKER_WORD)
            break
        return 0

    def _scan(self, final: bool) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        buf = self._buf

        while buf:
            marker_at = buf.find(SPEAKER_OPEN)
            word_at = self._find_speaker_word(buf)
            candidates = [i for i in (marker_at, word_at) if i != -1]

            if not candidates:
                keep = 0 if final else max(
                    _partial_suffix_len(buf, SPEAKER_OPEN), self._boundary_partial_len(buf)
                )
                cut = len(buf) - keep
                self._emit(buf[:cut], out)
                buf = buf[cut:]
                break

            index = min(candidates)

            if index == marker_at:
                rest = buf[index + len(SPEAKER_OPEN) :]
                close = rest.find(SPEAKER_CLOSE)
                if close == -1:
                    self._emit(buf[:index], out)
                    if final:
                        self._emit(clean_text(rest), out)
                        buf = ""
                    else:
                        buf = buf[index:]
                    break
                self._emit(buf[:index], out)
                self._new_segment(clean_text(rest[:close]).strip() or None, out)
                buf = rest[close + len(SPEAKER_CLOSE) :]
                continue

            match = _SPEAKER_FULL.match(buf, index)
            if match:
                self._emit(buf[:index], out)
                self._new_segment(match.group()[:-1].strip() or None, out)
                buf = buf[match.end() :]
                continue

            if not final and _SPEAKER_PARTIAL.match(buf, index):
                # Colon may still be on its way.
                self._emit(buf[:index], out)
                buf = buf[index:]
                break

            # A "Speaker" that is not a turn prefix: pass it through.
            self._emit(buf[: index + len(SPEAKER_WORD)], out)
            buf = buf[index + len(SPEAKER_WORD) :]

        self._buf = buf
        return out

    # -- segment bookkeeping ----------------------------------------------
    def _new_segment(self, speaker: Optional[str], out: list) -> None:
        if self.segments:
            self.segments[-1].text = self.segments[-1].text.rstrip()
        self.segments.append(
            Segment(speaker=speaker, chunk_start=self._chunk, chunk_end=self._chunk)
        )
        out.append((len(self.segments) - 1, ""))

    def _emit(self, text: str, out: list) -> None:
        text = clean_text(text)
        if not text:
            return
        if not self.segments:
            # Leading whitespace before the first turn is separator noise.
            if not text.strip():
                self._last_char = text[-1]
                return
            self.segments.append(Segment(speaker=None, chunk_start=self._chunk, chunk_end=self._chunk))
        current = self.segments[-1]
        if not current.text:
            text = text.lstrip()
            if not text:
                return
        current.text += text
        current.chunk_end = self._chunk
        self._last_char = text[-1]
        out.append((len(self.segments) - 1, text))

    def as_dicts(self) -> list[dict]:
        return [s.as_dict() for s in self.segments if s.text.strip()]


# --------------------------------------------------------------------------
# Model engine
# --------------------------------------------------------------------------


@dataclass
class FrameConfig:
    sample_rate: int
    frame_samples: int
    chunk_frames: int
    lookahead_frames: int

    @property
    def chunk_samples(self) -> int:
        return self.chunk_frames * self.frame_samples

    @property
    def lookahead_samples(self) -> int:
        return self.lookahead_frames * self.frame_samples

    @property
    def window_samples(self) -> int:
        return self.chunk_samples + self.lookahead_samples

    @property
    def chunk_seconds(self) -> float:
        return self.chunk_samples / self.sample_rate

    @property
    def lookahead_seconds(self) -> float:
        return self.lookahead_samples / self.sample_rate

    def as_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "frame_samples": self.frame_samples,
            "chunk_frames": self.chunk_frames,
            "lookahead_frames": self.lookahead_frames,
            "chunk_samples": self.chunk_samples,
            "window_samples": self.window_samples,
            "chunk_seconds": round(self.chunk_seconds, 4),
            "lookahead_seconds": round(self.lookahead_seconds, 4),
        }


@dataclass
class DecodeOptions:
    context_info: Optional[str] = None
    temperature: float = 0.0
    max_new_tokens: int = 256
    repetition_penalty: float = 1.0

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "DecodeOptions":
        raw = raw or {}
        ctx = raw.get("context_info")
        if isinstance(ctx, str):
            ctx = ctx.strip() or None
        return cls(
            context_info=ctx,
            temperature=max(0.0, min(2.0, float(raw.get("temperature") or 0.0))),
            max_new_tokens=max(16, min(1024, int(raw.get("max_new_tokens") or 256))),
            repetition_penalty=max(1.0, min(2.0, float(raw.get("repetition_penalty") or 1.0))),
        )


class IncrementalDecoder:
    """Decodes a growing id list into text deltas without splitting UTF-8."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self._emitted = ""

    def add(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=False)
        # A trailing replacement char means the last token is half a codepoint;
        # hold it back until the rest arrives.
        if text.endswith("�"):
            return ""
        if not text.startswith(self._emitted):
            # Non-monotonic decode (rare); resync on the common prefix.
            common = os.path.commonprefix([text, self._emitted])
            self._emitted = common
        piece = text[len(self._emitted) :]
        self._emitted = text
        return piece

    def finish(self) -> str:
        text = self.tokenizer.decode(self.ids, skip_special_tokens=False)
        piece = text[len(self._emitted) :] if text.startswith(self._emitted) else ""
        self._emitted = text
        return piece

    @property
    def text(self) -> str:
        """Everything decoded so far, special tokens included."""
        return self._emitted


class StreamingSession:
    """One transcription in flight: owns its KV cache and transcript state."""

    def __init__(self, engine: "VibeVoiceEngine", options: DecodeOptions) -> None:
        self.engine = engine
        self.options = options
        self.builder = TranscriptBuilder()
        self.chunk_index = 0
        self.tokens_generated = 0
        self.chunk_latencies_ms: list[float] = []
        self.first_token_latencies_ms: list[float] = []
        self.audio_samples = 0
        self.started = time.time()
        with engine.gpu_lock:
            self.state = engine.model.init_streaming_state(
                engine.tokenizer, context_info=options.context_info
            )

    # -- metrics ----------------------------------------------------------
    def stats(self) -> dict:
        audio_seconds = self.audio_samples / self.engine.frames.sample_rate
        elapsed = time.time() - self.started
        lat = self.chunk_latencies_ms
        ftl = self.first_token_latencies_ms
        # RTF is GPU time per second of audio, not wall clock: in live mode the
        # wall clock is pinned to real time by the microphone and would always
        # read ~1.0 no matter how fast the model actually is.
        compute_seconds = sum(lat) / 1000.0
        return {
            "chunks": self.chunk_index,
            "tokens": self.tokens_generated,
            "audio_seconds": round(audio_seconds, 2),
            "elapsed_seconds": round(elapsed, 2),
            "compute_seconds": round(compute_seconds, 2),
            "rtf": round(compute_seconds / audio_seconds, 3) if audio_seconds > 0 else None,
            "chunk_latency_ms_avg": round(sum(lat) / len(lat), 1) if lat else None,
            "chunk_latency_ms_last": round(lat[-1], 1) if lat else None,
            "first_token_ms_avg": round(sum(ftl) / len(ftl), 1) if ftl else None,
        }

    def result(self) -> dict:
        return {
            "text": self.builder.text.strip(),
            # Same words without the speaker labels. Dictation is one person
            # talking, and "Speaker 0:" would be pasted into their document.
            "plain": self.builder.plain_text.strip(),
            "raw": self.builder.raw,
            "segments": self.builder.as_dicts(),
            "stats": self.stats(),
        }

    # -- inference --------------------------------------------------------
    def process_window(self, window: np.ndarray, advance_samples: int, on_event: Callable[[dict], None]) -> None:
        """Encode one window (chunk + lookahead) and decode its text."""
        engine = self.engine
        self.builder.set_chunk(self.chunk_index)
        started = time.perf_counter()
        first_token_at: Optional[float] = None
        decoder = IncrementalDecoder(engine.tokenizer)

        def on_token(token_id: int) -> None:
            nonlocal first_token_at
            if first_token_at is None:
                first_token_at = time.perf_counter()
            self.tokens_generated += 1
            piece = decoder.add(token_id)
            for seg_index, appended in self.builder.feed(piece):
                on_event(
                    {
                        "type": "delta",
                        "segment": seg_index,
                        "speaker": self.builder.segments[seg_index].speaker,
                        "text": appended,
                        "chunk": self.chunk_index,
                    }
                )

        with engine.gpu_lock:
            audio = torch.from_numpy(np.ascontiguousarray(window)).to(engine.device)
            features = engine.model.encode_speech(audio.unsqueeze(0))
            engine._run_chunk(
                self.state,
                features,
                max_new_tokens=self.options.max_new_tokens,
                temperature=self.options.temperature,
                repetition_penalty=self.options.repetition_penalty,
                on_token=on_token,
            )

        tail = decoder.finish()
        if tail:
            for seg_index, appended in self.builder.feed(tail):
                on_event(
                    {
                        "type": "delta",
                        "segment": seg_index,
                        "speaker": self.builder.segments[seg_index].speaker,
                        "text": appended,
                        "chunk": self.chunk_index,
                    }
                )

        latency_ms = (time.perf_counter() - started) * 1000.0
        self.chunk_latencies_ms.append(latency_ms)
        if first_token_at is not None:
            self.first_token_latencies_ms.append((first_token_at - started) * 1000.0)
        self.audio_samples += advance_samples

        chunk_seconds = engine.frames.chunk_seconds
        on_event(
            {
                "type": "chunk",
                "index": self.chunk_index,
                "audio_start": round(self.chunk_index * chunk_seconds, 3),
                "audio_end": round((self.chunk_index + 1) * chunk_seconds, 3),
                "latency_ms": round(latency_ms, 1),
                "text": clean_text(decoder.text).strip(),
                "raw": decoder.text,
                "stats": self.stats(),
            }
        )
        self.chunk_index += 1

    def finalize(self, on_event: Callable[[dict], None]) -> dict:
        for seg_index, appended in self.builder.flush():
            on_event(
                {
                    "type": "delta",
                    "segment": seg_index,
                    "speaker": self.builder.segments[seg_index].speaker,
                    "text": appended,
                    "chunk": max(0, self.chunk_index - 1),
                }
            )
        payload = self.result()
        payload["type"] = "done"
        on_event(payload)
        return payload

    def release(self) -> None:
        self.state = None


class VibeVoiceEngine:
    """Holds the checkpoint; one instance per process."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.attn_implementation = attn_implementation
        self.dtype = dtype if device == "cuda" else torch.float32
        self.gpu_lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.frames: Optional[FrameConfig] = None
        self.load_seconds: Optional[float] = None

    # -- loading ----------------------------------------------------------
    def _read_frame_config(self) -> FrameConfig:
        path = os.path.join(self.model_path, "preprocessor_config.json")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"{path} not found. Mount the downloaded checkpoint at {self.model_path}."
            )
        with open(path) as fh:
            cfg = json.load(fh)
        missing = [k for k in ("chunk_frames", "lookahead_frames") if k not in cfg]
        if missing:
            raise RuntimeError(
                f"{path} has no {', '.join(missing)}: this is not a streaming checkpoint."
            )
        return FrameConfig(
            sample_rate=int(cfg["target_sample_rate"]),
            frame_samples=int(cfg["speech_tok_compress_ratio"]),
            chunk_frames=int(cfg["chunk_frames"]),
            lookahead_frames=int(cfg["lookahead_frames"]),
        )

    def load(self) -> None:
        started = time.time()
        self.frames = self._read_frame_config()
        self.processor = VibeVoiceASRProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        if getattr(self.tokenizer, "text_chunk_end_id", None) is None:
            raise RuntimeError(
                "Tokenizer has no <|text_chunk_end|>: this is not a streaming checkpoint."
            )
        # device_map streams the shards straight onto the GPU instead of
        # materialising 17 GB in host RAM first.
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            device_map=self.device,
        )
        self.model = model.eval()
        # encode_speech casts the waveform to config.torch_dtype; keep it in step
        # with the weights we actually loaded.
        try:
            self.model.config.torch_dtype = self.dtype
        except Exception:  # pragma: no cover - config is permissive in practice
            pass
        self.load_seconds = time.time() - started

    @property
    def ready(self) -> bool:
        return self.model is not None

    def info(self) -> dict:
        payload = {
            "model_path": self.model_path,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "attn_implementation": self.attn_implementation,
            "ready": self.ready,
            "load_seconds": round(self.load_seconds, 1) if self.load_seconds else None,
        }
        if self.frames:
            payload["frames"] = self.frames.as_dict()
        if torch.cuda.is_available():
            payload["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            }
        return payload

    # -- decoding ---------------------------------------------------------
    @torch.no_grad()
    def _run_chunk(
        self,
        state: dict,
        features: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
        on_token: Callable[[int], None],
    ) -> list[int]:
        """Mirror of upstream `streaming_generate_step`, but hands back raw ids."""
        model = self.model
        device = self.device
        embed_tokens = state["embed_tokens"]
        tce_id = state["text_chunk_end_id"]
        eos_id = state["eos_id"]

        audio_embeds = torch.cat(
            [state["sp_start_embed"], features.to(state["sp_start_embed"].dtype), state["sp_end_embed"]],
            dim=1,
        )
        outputs = model(
            inputs_embeds=audio_embeds,
            past_key_values=state["past_key_values"],
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        ids: list[int] = []
        for _ in range(max_new_tokens):
            step_logits = logits
            if repetition_penalty != 1.0 and ids:
                step_logits = step_logits.clone()
                prev = torch.tensor(sorted(set(ids)), device=step_logits.device)
                picked = step_logits[:, prev]
                step_logits[:, prev] = torch.where(
                    picked > 0, picked / repetition_penalty, picked * repetition_penalty
                )

            if temperature <= 0:
                token_id = int(torch.argmax(step_logits, dim=-1).item())
            else:
                probs = torch.softmax(step_logits.float() / temperature, dim=-1)
                token_id = int(torch.multinomial(probs, num_samples=1).squeeze(-1).item())

            if token_id == tce_id or token_id == eos_id:
                break

            ids.append(token_id)
            on_token(token_id)

            outputs = model(
                inputs_embeds=embed_tokens(torch.tensor([[token_id]], device=device)),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        # Close the chunk so the next one starts from a clean boundary.
        outputs = model(
            inputs_embeds=embed_tokens(torch.tensor([[tce_id]], device=device)),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        state["past_key_values"] = outputs.past_key_values
        return ids

    # -- public entry points ----------------------------------------------
    def new_session(self, options: DecodeOptions) -> StreamingSession:
        if not self.ready:
            raise RuntimeError("Model is still loading")
        return StreamingSession(self, options)

    def transcribe_array(
        self,
        audio: np.ndarray,
        options: DecodeOptions,
        on_event: Callable[[dict], None],
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """Run a complete waveform through the streaming loop."""
        frames = self.frames
        session = self.new_session(options)
        total = len(audio)
        chunk_samples = frames.chunk_samples
        window_samples = frames.window_samples
        total_chunks = max(1, (total + chunk_samples - 1) // chunk_samples)

        on_event(
            {
                "type": "meta",
                "duration": round(total / frames.sample_rate, 2),
                "total_chunks": total_chunks,
                "frames": frames.as_dict(),
                "options": {
                    "context_info": options.context_info,
                    "temperature": options.temperature,
                    "max_new_tokens": options.max_new_tokens,
                    "repetition_penalty": options.repetition_penalty,
                },
            }
        )

        start = 0
        try:
            while start < total:
                if should_stop is not None and should_stop():
                    break
                end = min(start + window_samples, total)
                window = audio[start:end]
                if len(window) < window_samples:
                    padded = np.zeros(window_samples, dtype=np.float32)
                    padded[: len(window)] = window
                    window = padded
                advance = min(chunk_samples, total - start)
                session.process_window(window, advance, on_event)
                start += chunk_samples
            return session.finalize(on_event)
        finally:
            session.release()
