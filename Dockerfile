# FeatherBound vision cloud tier — BioCLIP 2 species ID over ~11k birds (CPU).
FROM python:3.11-slim
WORKDIR /app
ENV HF_HOME=/app/hfcache HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir open_clip_torch fastapi "uvicorn[standard]" pillow numpy python-multipart

COPY bird_bank.npz app.py ./
# Bake the BioCLIP-2 weights into the image so the first request is instant (no runtime download).
RUN python -c "import open_clip; open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')"

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
