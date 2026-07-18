"""FeatherBound vision cloud tier.
- POST /identify   (BioCLIP 2 embedding ID over ~11k birds) -> top-k {sci, score}
- POST /gemini-id  (general VLM read for HARD/blurry photos the on-device model + BioCLIP miss)
Both bearer-key protected. Only hit when the on-device ensemble is unsure / regionally implausible."""
import io, os, re, json, base64, urllib.request, urllib.error
import numpy as np
import torch
import open_clip
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query, Form
from PIL import Image

MODEL = "hf-hub:imageomics/bioclip-2"
API_KEY = os.environ.get("VISION_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
model = model.eval()
_bank = np.load("bird_bank.npz", allow_pickle=True)
EMB = _bank["emb"].astype("float32")           # (N, 768) L2-normalized
LABELS = [str(x) for x in _bank["labels"]]     # eBird scientific names

app = FastAPI(title="FeatherBound Vision")


@app.get("/health")
def health():
    return {"ok": True, "species": len(LABELS), "model": MODEL,
            "gemini": bool(GEMINI_API_KEY), "gemini_model": GEMINI_MODEL}


GEMINI_PROMPT = (
    "You are an expert field ornithologist. Identify the bird in this photo. "
    "The image may be blurry, distant, or low quality. Based only on visible field marks, "
    "give your top 3 most likely species. For each line use EXACTLY this format:\n"
    "1. Common Name (Scientific name) - NN% - short reason from visible marks\n"
    "If you genuinely cannot tell, still give your 3 best guesses."
)
_LINE = re.compile(r"^\s*\d+[\.\)]\s*(.+?)\s*\(([^)]+)\)\s*[-–]\s*(\d+)\s*%?\s*[-–]\s*(.+?)\s*$")


def _resize_b64(raw: bytes, maxside: int = 1024) -> str:
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    if max(w, h) > maxside:
        sc = maxside / max(w, h)
        im = im.resize((int(w * sc), int(h * sc)))
    b = io.BytesIO(); im.save(b, "JPEG", quality=90)
    return base64.b64encode(b.getvalue()).decode()


def _demd(s: str) -> str:
    return s.replace("*", "").replace("_", "").strip()


def _parse_gemini(text: str):
    out = []
    for ln in text.splitlines():
        m = _LINE.match(ln)
        if m:
            out.append({"common": _demd(m.group(1)), "sci": _demd(m.group(2)),
                        "confidence": int(m.group(3)), "reason": m.group(4).strip()})
    return out


@app.post("/gemini-id")
async def gemini_id(
    file: UploadFile = File(...),
    region: str = Form(default=""),        # optional: plausible-species hint or region name
    authorization: str = Header(default=""),
):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="gemini not configured")
    raw = await file.read()
    try:
        data = _resize_b64(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="bad image")
    prompt = GEMINI_PROMPT
    if region.strip():
        prompt += f"\nContext: the photo was taken in/near {region.strip()} — prefer species that occur there, but a clear out-of-range ID is allowed."
    body = json.dumps({
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": data}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400,
                             "thinkingConfig": {"thinkingBudget": 0}},
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"gemini {e.code}")
    except Exception:
        raise HTTPException(status_code=504, detail="gemini timeout")
    txt = ""
    for p in resp.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        if "text" in p:
            txt += p["text"]
    return {"results": _parse_gemini(txt), "raw": txt.strip(), "model": GEMINI_MODEL}


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
