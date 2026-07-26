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

# ---- Merlin: TEST-ONLY comparison judge (Cornell's model_v55, extracted from the APK). Active ONLY
# when BOTH the litert runtime AND /app/merlin_v55.tflite are present, so it ships NOWHERE by default.
# It is OBSERVATIONAL: logged + returned for comparison but never fed into the consensus, so removing
# it before launch is a no-op. (Legal: Cornell's model — internal benchmarking only, not the product.)
MERLIN = None
MERLIN_SCI = []
try:
    if os.path.exists("/app/merlin_v55.tflite"):
        from ai_edge_litert.interpreter import Interpreter as _LiteRT
        MERLIN = _LiteRT(model_path="/app/merlin_v55.tflite", num_threads=max(1, (os.cpu_count() or 2) - 1))
        MERLIN.allocate_tensors()
        MERLIN_SCI = [ln.strip() for ln in open("/app/merlin_sci.txt", encoding="utf-8")]
        print(f"MERLIN judge loaded: {len(MERLIN_SCI)} classes", flush=True)
except Exception as e:
    print("MERLIN load failed (judge disabled): " + repr(e), flush=True)
    MERLIN = None


def _merlin_topk(img, k=8, lat=None, lng=None):
    """Merlin's own top-k. Recipe from the head-to-head: resize min-side to 224, center-crop, raw
    float32 0-255 (the model normalizes internally); model_v55 already outputs softmax probs."""
    if MERLIN is None:
        return None
    try:
        w, h = img.size
        s = 224.0 / min(w, h)
        im = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
        w, h = im.size
        l, t = (w - 224) // 2, (h - 224) // 2
        arr = np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32)
        inp = MERLIN.get_input_details()[0]
        outd = MERLIN.get_output_details()[0]
        MERLIN.set_tensor(inp["index"], arr[None])
        MERLIN.invoke()
        p = MERLIN.get_tensor(outd["index"])[0].astype("float32").copy()   # model_v55 already outputs softmax probs
        # Range prior: soft multiplicative down-weight for out-of-range species (Mike: even across judges).
        ok = _range_ok(MERLIN_ROWS, lat, lng)
        p = p * np.where(ok, 1.0, RANGE_OUT_FACTOR).astype("float32")
        top = np.argsort(p)[-k:][::-1]
        return [{"sci": MERLIN_SCI[i], "score": float(p[i])} for i in top if i < len(MERLIN_SCI)]
    except Exception as ex:
        print("MERLIN infer failed: " + repr(ex), flush=True)
        return None


# Detector-crop so the Merlin judge gets a tight bird crop (it reaches its real accuracy on crops,
# not full scenes — head-to-head finding). torchvision COCO fasterrcnn, bird = class 16. Loaded only
# alongside Merlin (test-only infra). BioCLIP keeps the full scene (it prefers it), so each model
# gets its preferred input for a fair head-to-head.
MERLIN_DETECTOR = None
if MERLIN is not None:
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        MERLIN_DETECTOR = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
        print("MERLIN detector loaded", flush=True)
    except Exception as e:
        print("MERLIN detector load failed: " + repr(e), flush=True)
        MERLIN_DETECTOR = None


def _bird_crop(img):
    """Highest-confidence COCO 'bird' box (+15% pad); None if no bird found -> Merlin center-crops."""
    if MERLIN_DETECTOR is None:
        return None
    try:
        x = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0)
        with torch.no_grad():
            out = MERLIN_DETECTOR([x])[0]
        boxes, labels, scores = out["boxes"].numpy(), out["labels"].numpy(), out["scores"].numpy()
        birds = [(b, s) for b, l, s in zip(boxes, labels, scores) if l == 16 and s > 0.3]
        if not birds:
            return None
        b = max(birds, key=lambda bs: bs[1])[0]
        w, h = img.size
        x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        pw, ph = (x1 - x0) * 0.15, (y1 - y0) * 0.15
        return img.crop((int(max(0, x0 - pw)), int(max(0, y0 - ph)),
                         int(min(w, x1 + pw)), int(min(h, y1 + ph))))
    except Exception as ex:
        print("MERLIN detector infer failed: " + repr(ex), flush=True)
        return None


# ---- Range prior (GBIF): give EVERY judge the location signal so the panel is "even" (Mike). It is
# a SOFT down-weight of out-of-range species, never a hard filter — vagrants/zoo birds stay reachable,
# and species with no GBIF range data default to in-range. Applied to on-device, BioCLIP and Merlin;
# Gemini already gets the location in its prompt. Loads only if range_grid.npz is present. ----
RANGE_GRID = None
RANGE_HAS = None
RANGE_HW = (0, 0)
RANGE_SCI2ROW = {}
RANGE_OUT_FACTOR = 0.05
LABEL_ROWS = np.full(len(LABELS), -1, np.int64)
MERLIN_ROWS = np.full(len(MERLIN_SCI), -1, np.int64) if MERLIN_SCI else np.zeros(0, np.int64)
try:
    if os.path.exists("range_grid.npz"):
        _z = np.load("range_grid.npz", allow_pickle=True)
        RANGE_HW = (int(_z["shape"][0]), int(_z["shape"][1]))
        RANGE_GRID = np.unpackbits(_z["packed"], axis=1).reshape(-1, RANGE_HW[0], RANGE_HW[1]).astype(bool)
        RANGE_HAS = _z["has_data"].astype(bool)
        _rsci = [ln.strip() for ln in open("range_species.txt", encoding="utf-8")]
        RANGE_SCI2ROW = {s: i for i, s in enumerate(_rsci)}
        LABEL_ROWS = np.array([RANGE_SCI2ROW.get(s, -1) for s in LABELS], np.int64)
        if MERLIN_SCI:
            MERLIN_ROWS = np.array([RANGE_SCI2ROW.get(s, -1) for s in MERLIN_SCI], np.int64)
        print(f"RANGE prior loaded: {len(_rsci)} species {RANGE_HW[0]}x{RANGE_HW[1]}", flush=True)
except Exception as e:
    print("RANGE load failed (prior disabled): " + repr(e), flush=True)
    RANGE_GRID = None


def _range_ok(rows, lat, lng):
    """Boolean per-row: True = in range (or unknown species / no location / no grid)."""
    if RANGE_GRID is None or lat is None or lng is None:
        return np.ones(len(rows), bool)
    gH, gW = RANGE_HW
    r = min(gH - 1, max(0, int((90 - lat) / 180 * gH)))
    c = min(gW - 1, max(0, int((lng + 180) / 360 * gW)))
    ok = np.ones(len(rows), bool)
    valid = rows >= 0
    vr = rows[valid]
    ok[valid] = RANGE_GRID[vr, r, c] | ~RANGE_HAS[vr]   # present in cell OR no range data -> keep
    return ok

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


def _bioclip_topk(img, k=8, lat=None, lng=None):
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        ie = model.encode_image(x)
        ie = (ie / ie.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")
    sims = (ie @ EMB.T)[0].copy()
    # Range prior: soft cosine penalty for species out of range at this location (never a hard filter).
    ok = _range_ok(LABEL_ROWS, lat, lng)
    sims = sims - np.where(ok, 0.0, 0.12).astype("float32")
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
    merlin: str = Form(default=""),       # "1" to also run the slow test-only Merlin judge (off by default)
    authorization: str = Header(default=""),
):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="bad image")

    def _f(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return None
    latf, lngf = _f(lat), _f(lng)

    # Judge A: on-device model (its top-K, handed up from the phone) — range-adjusted so it's even too
    try:
        od = json.loads(ondevice) or []
    except Exception:
        od = []
    od = [d for d in od if isinstance(d, dict) and d.get("sci")]
    if latf is not None and lngf is not None and od:
        od_ok = _range_ok(np.array([RANGE_SCI2ROW.get(str(d["sci"]), -1) for d in od], np.int64), latf, lngf)
        for d, okv in zip(od, od_ok):
            if not okv:
                d["confidence"] = float(d.get("confidence", 0) or 0) * RANGE_OUT_FACTOR
        od.sort(key=lambda d: -float(d.get("confidence", 0) or 0))
    od_scis = [str(d["sci"]) for d in od]

    # Judge B: our hosted BioCLIP-2 (range-adjusted)
    bio = _bioclip_topk(img, k=8, lat=latf, lng=lngf)
    bio_scis = [d["sci"] for d in bio]

    # Judge D (TEST-ONLY, observational): Merlin. It adds a slow CPU detector-crop (~2s) and is NOT
    # in the consensus, so it runs ONLY when explicitly requested (`merlin=1`). The app skips it, which
    # keeps the FeatherBound+ call fast (BioCLIP + Gemini only).
    mcrop = None
    mer = None
    if merlin.strip() in ("1", "true", "yes"):
        mcrop = _bird_crop(img)
        mer = _merlin_topk(mcrop or img, k=8, lat=latf, lng=lngf)
    mer_scis = [d["sci"] for d in mer] if mer else []

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
              "judges": {"ondevice": od[:8], "bioclip": bio, "merlin": mer, "gemini": gem}}
    print("PANEL " + json.dumps({"consensus": consensus, "region": region, "date": date,
                                 "ondevice_top": od_scis[:1], "bioclip_top": bio_scis[:1],
                                 "merlin_top": mer_scis[:1], "merlin_cropped": mcrop is not None,
                                 "gemini": (gem or {}).get("sci")}), flush=True)
    return result
