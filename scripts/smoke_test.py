#!/usr/bin/env python3
"""End-to-end check of the backend without a browser.

    python3 scripts/smoke_test.py                       # health + config + samples
    python3 scripts/smoke_test.py --file audio.mp3      # SSE file transcription
    python3 scripts/smoke_test.py --live samples/demo1-chat.mp3
                                                        # stream it over the live
                                                        # websocket at real-time pace

Needs only the stdlib plus `websockets` for --live.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import time
import urllib.request
import uuid

BASE = os.environ.get("VIBEVOICE_API", "http://localhost:8001")


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=30) as res:
        return json.loads(res.read())


def post_multipart(path: str, fields: dict, file_path: str):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for key, value in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
    mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    with open(file_path, "rb") as fh:
        payload = fh.read()
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{os.path.basename(file_path)}\"\r\nContent-Type: {mime}\r\n\r\n"
    ).encode()
    body += payload + f"\r\n--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        BASE + path,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return urllib.request.urlopen(request, timeout=3600)


def stream_file(path: str, hotwords: str, temperature: float):
    print(f"POST /api/transcribe  ({os.path.basename(path)})")
    started = time.time()
    response = post_multipart(
        "/api/transcribe",
        {"context_info": hotwords, "temperature": str(temperature), "max_new_tokens": "256"},
        path,
    )
    final = None
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        event = json.loads(line[5:].strip())
        kind = event.get("type")
        if kind == "meta":
            print(f"  audio {event['duration']}s in {event['total_chunks']} chunks "
                  f"of {event['frames']['chunk_seconds']}s")
        elif kind == "chunk":
            print(f"  [{event['index'] + 1}] {event['latency_ms']:.0f}ms  {event['text']!r}")
        elif kind == "done":
            final = event
        elif kind == "error":
            print("  ERROR:", event["message"])
            return None
    if final:
        print("\n--- transcript ---")
        for seg in final["segments"]:
            print(f"  {seg['speaker'] or '(unattributed)'}: {seg['text'].strip()}")
        print("\n--- raw ---")
        print(" ", final["raw"][:600].replace("\n", "\\n"))
        print("\n--- stats ---", json.dumps(final["stats"]))
        print(f"wall clock {time.time() - started:.1f}s")
    return final


def stream_live(path: str, hotwords: str, realtime: bool):
    import asyncio

    import websockets

    sample_rate = get("/api/config")["frames"]["sample_rate"]
    pcm = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-nostdin", "-i", path,
         "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    total = len(pcm) // 4
    print(f"streaming {total / sample_rate:.1f}s of audio over /api/ws/live"
          f"{' at real-time pace' if realtime else ' as fast as possible'}")

    async def run():
        url = BASE.replace("http://", "ws://").replace("https://", "wss://") + "/api/ws/live"
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(json.dumps({"context_info": hotwords or None, "temperature": 0.0}))

            async def reader():
                async for message in ws:
                    event = json.loads(message)
                    if event["type"] == "chunk":
                        print(f"  [{event['index'] + 1}] {event['latency_ms']:.0f}ms  {event['text']!r}")
                    elif event["type"] == "done":
                        print("\n--- transcript ---")
                        for seg in event["segments"]:
                            print(f"  {seg['speaker'] or '(unattributed)'}: {seg['text'].strip()}")
                        print("--- stats ---", json.dumps(event["stats"]))
                        return
                    elif event["type"] == "error":
                        print("  ERROR:", event["message"])
                        return

            task = asyncio.create_task(reader())
            step = sample_rate // 10  # 100 ms packets, like a browser
            for offset in range(0, total, step):
                await ws.send(pcm[offset * 4 : (offset + step) * 4])
                if realtime:
                    await asyncio.sleep(step / sample_rate)
            await ws.send("end")
            await asyncio.wait_for(task, timeout=900)

    asyncio.run(run())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", help="transcribe this file over SSE")
    parser.add_argument("--live", help="stream this file over the live websocket")
    parser.add_argument("--hotwords", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--asap", action="store_true", help="with --live, do not pace to real time")
    args = parser.parse_args()

    health = get("/api/health")
    print("health:", json.dumps({k: health[k] for k in ("status", "device", "dtype", "attn_implementation") if k in health}))
    if health.get("gpu"):
        print("gpu   :", json.dumps(health["gpu"]))
    if health["status"] != "ready":
        print("model is not ready yet:", health.get("error") or "still loading")
        return 1

    print("config:", json.dumps(get("/api/config")["frames"]))
    print("samples:", [s["id"] for s in get("/api/samples")])

    if args.file:
        stream_file(args.file, args.hotwords, args.temperature)
    if args.live:
        stream_live(args.live, args.hotwords, not args.asap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
