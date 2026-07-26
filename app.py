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
# A /app/gemini_key.txt (if present) wins over the env var, so the Gemini key can be hot-swapped
# with just a container restart (no recreate / no model re-download) if the env key gets capped.
def _gemini_key():
    try:
        with open("/app/gemini_key.txt") as f:
            k = f.read().strip()
            if k:
                return k
    except OSError:
        pass
    return os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_KEY = _gemini_key()
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


# ---- Panel of judges: fuse on-device + BioCLIP-2 + Gemini into one consensus ----
# The premium "cloud brain": the phone sends its own top-K plus the photo + location/date; we run
# our hosted BioCLIP-2, fuse the two ranked lists, and let Gemini adjudicate the shortlist (with an
# escape hatch to override when both classifiers miss). Every judge is logged so we can compare.
# (Merlin joins as a 4th, test-only judge in a follow-up once its model file is wired in.)

_LABEL_LC = {s.lower(): s for s in LABELS}   # sci (lowercased) -> canonical sci, for validating a VLM pick


def _bioclip_topk(img, k=8):
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        ie = model.encode_image(x)
        ie = (ie / ie.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")
    sims = (ie @ EMB.T)[0]
    top = np.argsort(sims)[-k:][::-1]
    return [{"sci": LABELS[i], "score": float(sims[i])} for i in top]


def _rrf(rank_lists, kk: int = 60):
    """Reciprocal-rank fusion across ranked sci-name lists -> {sci: fused_score}. Scale-free, so it
    combines the phone's softmax shares and BioCLIP's cosine sims without needing comparable units."""
    scores = {}
    for lst in rank_lists:
        for rank, sci in enumerate(lst):
            scores[sci] = scores.get(sci, 0.0) + 1.0 / (kk + rank + 1)
    return scores


_PANEL_PROMPT = (
    "You are an expert field ornithologist adjudicating a bird photo. Two image classifiers proposed "
    "this shortlist of candidate species (scientific names):\n{shortlist}\n{context}"
    "Decide which single species the bird most likely is. Strongly prefer a species from the shortlist. "
    "Only if NONE of them fit the visible field marks, name the correct species yourself. "
    "Reply on ONE line, EXACTLY in this format:\n"
    "Common Name (Scientific name) - NN% - short reason from visible marks"
)
_ONE = re.compile(r"^\s*(.+?)\s*\(([^)]+)\)\s*[-–]\s*(\d+)\s*%?\s*[-–]\s*(.+?)\s*$")


def _gemini_adjudicate(img_b64: str, shortlist_sci, context: str):
    prompt = _PANEL_PROMPT.format(
        shortlist="\n".join(f"- {s}" for s in shortlist_sci),
        context=(context + "\n") if context else "")
    body = json.dumps({
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 200,
                             "thinkingConfig": {"thinkingBudget": 0}},
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    txt = ""
    for p in resp.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        if "text" in p:
            txt += p["text"]
    for ln in txt.splitlines():
        m = _ONE.match(ln)
        if m:
            return {"common": _demd(m.group(1)), "sci": _demd(m.group(2)),
                    "confidence": int(m.group(3)), "reason": m.group(4).strip(), "raw": txt.strip()}
    return {"raw": txt.strip()}


@app.post("/panel")
async def panel(
    file: UploadFile = File(...),
    ondevice: str = Form(default="[]"),   # JSON: [{"sci": "...", "confidence": 0.42, "common": "..."}]
    lat: str = Form(default=""),
    lng: str = Form(default=""),
    date: str = Form(default=""),
    region: str = Form(default=""),       # human place name, optional
    k: int = Form(default=5),
    authorization: str = Header(default=""),
):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="bad image")

    # Judge A: on-device model (its top-K, handed up from the phone)
    try:
        od = json.loads(ondevice) or []
    except Exception:
        od = []
    od = [d for d in od if isinstance(d, dict) and d.get("sci")]
    od_scis = [str(d["sci"]) for d in od]

    # Judge B: our hosted BioCLIP-2
    bio = _bioclip_topk(img, k=8)
    bio_scis = [d["sci"] for d in bio]

    # Fuse the two ranked lists into a shortlist (reciprocal-rank fusion)
    fused = _rrf([od_scis, bio_scis])
    shortlist = [s for s, _ in sorted(fused.items(), key=lambda kv: -kv[1])][:max(k, 5)]

    # Judge C: Gemini adjudicates the shortlist, with an escape hatch to override it
    gem = None
    if GEMINI_API_KEY and shortlist:
        ctx = []
        if region.strip():
            ctx.append(f"The photo was taken in/near {region.strip()}.")
        elif lat.strip() and lng.strip():
            ctx.append(f"The photo was taken near {lat.strip()}, {lng.strip()}.")
        if date.strip():
            ctx.append(f"Date: {date.strip()}.")
        try:
            gem = _gemini_adjudicate(_resize_b64(raw), shortlist, " ".join(ctx))
        except Exception:
            gem = None

    # Consensus: Gemini's pick if valid; agree vs override depending on whether it stayed in the shortlist
    consensus = None
    sl_lc = {s.lower() for s in shortlist}
    if gem and gem.get("sci"):
        gsci_lc = gem["sci"].lower()
        canon = _LABEL_LC.get(gsci_lc)
        if gsci_lc in sl_lc:
            consensus = {"sci": canon or gem["sci"], "common": gem.get("common"),
                         "confidence": gem.get("confidence", 80), "source": "agree",
                         "reason": gem.get("reason")}
        elif canon:
            consensus = {"sci": canon, "common": gem.get("common"),
                         "confidence": gem.get("confidence", 70), "source": "gemini_override",
                         "reason": gem.get("reason")}
    if consensus is None:
        top_sci = shortlist[0] if shortlist else (od_scis[0] if od_scis else (bio_scis[0] if bio_scis else None))
        consensus = {"sci": top_sci, "common": None, "confidence": None,
                     "source": "classifier", "reason": None}

    result = {"consensus": consensus, "shortlist": shortlist,
              "judges": {"ondevice": od[:8], "bioclip": bio, "gemini": gem}}
    print("PANEL " + json.dumps({"consensus": consensus, "region": region, "date": date,
                                 "ondevice_top": od_scis[:1], "bioclip_top": bio_scis[:1],
                                 "gemini": (gem or {}).get("sci")}), flush=True)
    return result
