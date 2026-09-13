# Backend: FastAPI wrapper around microsoft/VibeVoice-ASR-Streaming-7B.
# The checkpoint is NOT baked into the image -- it is bind-mounted at /models,
# so rebuilding the app never re-copies 17 GB of weights.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/srv \
    MODEL_PATH=/models/VibeVoice-ASR-Streaming-7B \
    SAMPLES_DIR=/samples \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ffmpeg decodes every upload; curl is only there for the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements-torch.txt .
RUN pip install -r requirements-torch.txt

COPY requirements.txt .
RUN pip install -r requirements.txt

# Vendored from github.com/microsoft/VibeVoice (MIT) -- see vendor/LICENSE.vibevoice.
COPY vendor/vibevoice /srv/vibevoice
COPY app /srv/app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=200 \
  CMD curl -fsS http://127.0.0.1:8000/api/health >/dev/null || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "33554432"]
