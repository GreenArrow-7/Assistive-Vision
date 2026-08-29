# Assistive Vision — server image (works on Hugging Face Spaces / any Docker host)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image ~2GB smaller than default CUDA wheels
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# HF Spaces runs containers as uid 1000, not root. Model caches (~/.EasyOCR,
# the ultralytics config dir, matplotlib) all key off HOME — bake and run as
# the same user or the runtime re-downloads ~700MB into an unwritable path.
RUN useradd -m -u 1000 appuser && chown appuser:appuser /app
COPY --chown=appuser:appuser . .
USER appuser
ENV HOME=/home/appuser \
    YOLO_CONFIG_DIR=/home/appuser/.config/ultralytics \
    MPLCONFIGDIR=/tmp/mpl

# Bake models into the image so cold starts don't re-download
RUN python scripts/download_models.py

# HF Spaces expects 7860; override with -e PORT for other hosts
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT}"]
