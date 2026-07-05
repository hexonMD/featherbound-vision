# FeatherBound vision cloud tier — BioCLIP 2 species ID over ~11k birds (CPU).
FROM python:3.11-slim
WORKDIR /app
ENV HF_HOME=/app/hfcache HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir open_clip_torch fastapi "uvicorn[standard]" pillow numpy python-multipart

COPY bird_bank.npz app.py ./
# Model weights (~1.7GB) are NOT baked into the image (keeps it small on a disk-tight VM);
# they download on first startup into HF_HOME, which is a persistent Coolify volume.

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
