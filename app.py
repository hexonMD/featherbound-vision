"""FeatherBound vision cloud tier — BioCLIP 2 species ID over all ~11k birds.
Only hit when the on-device ensemble is unsure / the pick is regionally implausible.
POST /identify (multipart image) -> top-k {sci, score}. Bearer-key protected."""
import io, os
import numpy as np
import torch
import open_clip
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query
from PIL import Image

MODEL = "hf-hub:imageomics/bioclip-2"
API_KEY = os.environ.get("VISION_API_KEY", "")
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
model = model.eval()
_bank = np.load("bird_bank.npz", allow_pickle=True)
EMB = _bank["emb"].astype("float32")           # (N, 768) L2-normalized
LABELS = [str(x) for x in _bank["labels"]]     # eBird scientific names

app = FastAPI(title="FeatherBound Vision")


@app.get("/health")
def health():
    return {"ok": True, "species": len(LABELS), "model": MODEL}


@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=25),
    authorization: str = Header(default=""),
):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="bad image")
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        ie = model.encode_image(x)
        ie = (ie / ie.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")
    sims = (ie @ EMB.T)[0]
    top = np.argsort(sims)[-k:][::-1]
    return {"results": [{"sci": LABELS[i], "score": float(sims[i])} for i in top]}
