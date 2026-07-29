"""FeatherBound vision cloud tier.
- POST /identify   (BioCLIP 2 embedding ID over ~11k birds) -> top-k {sci, score}
- POST /gemini-id  (general VLM read for HARD/blurry photos the on-device model + BioCLIP miss)
Both bearer-key protected. Only hit when the on-device ensemble is unsure / regionally implausible."""
import io, os, re, json, base64, time, uuid, threading, urllib.request, urllib.error
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
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")          # cheap default judge
GEMINI_ESCALATE_MODEL = os.environ.get("GEMINI_ESCALATE_MODEL", "gemini-3.6-flash")  # strong judge (the app's model), hard cases
# Cost gate: when the on-device model's top pick is at least this confident, the cheap model confirms it
# (it's reliable on unmistakable birds); below it we spend the strong model on the genuinely hard birds.
PANEL_CONFIDENT_CONF = float(os.environ.get("PANEL_CONFIDENT_CONF", "0.70"))
# Below this Gemini confidence the grounded pass is "unsure" → fire the agentic code-execution re-check.
IDENTIFY_UNSURE_CONF = float(os.environ.get("IDENTIFY_UNSURE_CONF", "70"))
# Agentic re-check is OFF by default: it works but costs ~32s/call (code_execution loop), which exceeds
# the CF/proxy gateway timeout (504) synchronously. Ship it async first (grounded result instant, agentic
# refines the card later), then flip AGENTIC_FALLBACK=1. Flag kept so the capability is one env away.
AGENTIC_FALLBACK = os.environ.get("AGENTIC_FALLBACK", "0") == "1"
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
try:  # loaded ALWAYS now: Gemini needs a tight bird crop to read the bill on look-alikes (Merlin uses it too)
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    MERLIN_DETECTOR = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
    print("bird detector loaded (crop for Gemini + Merlin)", flush=True)
except Exception as e:
    print("bird detector load failed: " + repr(e), flush=True)
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


# Bird-only gate: the BioCLIP bank IS the world's birds, so its labels double as our "is this a
# bird" whitelist (genus-level, so a valid bird not in the bank still passes). Gemini is a general
# VLM that will name a lemur ("Indri indri") or a deer as a species — the panel now surfaces its
# off-catalog picks, so a non-bird could reach the user. We drop anything whose genus isn't a known
# bird genus, so only birds are ever shown.
_BIRD_SCI = set(LABELS)
_BIRD_GENERA = {l.split(" ")[0] for l in LABELS if l}
def _is_bird(sci: str) -> bool:
    sci = (sci or "").strip()
    if sci in _BIRD_SCI:
        return True
    return (sci.split(" ")[0] if sci else "") in _BIRD_GENERA


GEMINI_PROMPT = (
    "You are an expert field ornithologist. FIRST decide whether the main subject of the photo is a BIRD. "
    "If the subject is NOT a bird — for example a mammal, primate, lemur, reptile, amphibian, fish, insect, "
    "person, plant, or object — respond with EXACTLY one line and nothing else:\n"
    "NOT_A_BIRD - <what it actually is>\n"
    "Only if the subject IS a bird, identify it. The image may be blurry, distant, or low quality. Based only "
    "on visible field marks, give your top 3 most likely BIRD species. For each line use EXACTLY this format:\n"
    "1. Common Name (Scientific name) - NN% - short reason from visible marks\n"
    "If it is a bird but you genuinely cannot tell the species, still give your 3 best bird guesses. "
    "Never name a non-bird species."
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


def _pil_b64(im, maxside: int = 1100) -> str:
    im = im.convert("RGB"); w, h = im.size
    if max(w, h) > maxside:
        sc = maxside / max(w, h); im = im.resize((int(w * sc), int(h * sc)))
    b = io.BytesIO(); im.save(b, "JPEG", quality=93)
    return base64.b64encode(b.getvalue()).decode()


def _crop_b64(img):
    """Tight bird crop (detector) before encoding for Gemini — it needs the bill/eye at full spatial
    resolution to separate look-alikes; on a wide shot it misreads the bill and defaults to the common
    species. Falls back to the whole image when no bird box is found."""
    return _pil_b64(_bird_crop(img) or img)


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
    parsed = _parse_gemini(txt)
    # Bird-only gate: keep only bird species. If Gemini named a non-bird (e.g. "Indri indri", a
    # lemur), it's dropped; if that leaves nothing OR Gemini said NOT_A_BIRD, return an empty set
    # so the app shows "not a bird" instead of a bogus find.
    birds = [r for r in parsed if _is_bird(r.get("sci", ""))]
    non_bird = ("NOT_A_BIRD" in txt.upper()) or (bool(parsed) and not birds)
    return {"results": birds, "non_bird": non_bird, "raw": txt.strip(), "model": GEMINI_MODEL}


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
    "You are an expert field ornithologist. Identify the bird in this photo to species. Reason through "
    "these steps FIRST, then give the verdict:\n{context}"
    "1. BILL: describe its shape precisely — dagger-like/straight vs heavy/deep/arched/laterally-"
    "compressed vs slender/decurved/keeled. Then note tail shape+length, body plumage, and eye colour.\n"
    "2. The on-device classifiers guessed: {shortlist} — treat these as a hint only, not the answer.\n"
    "3. Compare the 2-3 species most consistent with those exact field marks AT THIS LOCATION "
    "(including but NOT limited to the guesses) and explicitly rule out the wrong ones by their bill/"
    "plumage — do not default to the commonest species if the bill says otherwise.\n"
    "4. On the LAST line, alone, EXACTLY in this format (no other text on that line):\n"
    "FINAL: Common Name (Scientific name) - NN% - short reason from the field marks"
)
_ONE = re.compile(r"^\s*(.+?)\s*\(([^)]+)\)\s*[-–]\s*(\d+)\s*%?\s*[-–]\s*(.+?)\s*$")


def _gemini_adjudicate(img_b64: str, shortlist_sci, context: str, model: str = None, thinking: int = -1):
    prompt = _PANEL_PROMPT.format(
        shortlist="\n".join(f"- {s}" for s in shortlist_sci),
        context=(context + "\n") if context else "")
    body = json.dumps({
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 3500,
                             "thinkingConfig": {"thinkingBudget": thinking}},
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model or GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    txt = ""
    for p in resp.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        if "text" in p:
            txt += p["text"]
    lines = txt.splitlines()
    # Prefer the explicit "FINAL:" verdict line; fall back to any parseable line.
    ordered = [l for l in lines if l.strip().upper().startswith("FINAL")] + lines
    for ln in ordered:
        cand = ln.strip()
        if cand.upper().startswith("FINAL"):
            cand = cand[5:].lstrip(": ").strip()
        m = _ONE.match(cand)
        if m:
            return {"common": _demd(m.group(1)), "sci": _demd(m.group(2)),
                    "confidence": int(m.group(3)), "reason": m.group(4).strip(), "raw": txt.strip()}
    return {"raw": txt.strip()}


_IDENTIFY_PROMPT = (
    "You are an expert field ornithologist. Identify the bird in this photo to a single species.\n"
    "{context}"
    "A bird-recognition model looked at THIS photo; its closest visual matches were:\n{shortlist}\n"
    "The true bird is most likely one of these or a close relative. Work through it:\n"
    "1. Describe ONLY what you can actually see — BILL depth/shape (the most diagnostic feature), tail, body "
    "proportions, posture, plumage. If the bird is blurry, small, or distant, say which marks are unclear; do "
    "NOT state a field mark you cannot clearly see.\n"
    "2. Compare the candidates above against those marks and pick the best fit. You may name a species NOT in "
    "the list ONLY if you can cite specific visible marks that rule out every candidate — do not switch to a "
    "locally-common species on a hunch, and never invent a bill or plumage you can't see.\n"
    "3. If the photo is too unclear to be confident, still give your best guess but a LOW confidence (<=55).\n"
    "Finish with a line, alone, EXACTLY in this format (no other text on that line):\n"
    "FINAL: Common Name (Scientific name) - NN% - short reason from the visible marks"
)


def _gemini_identify(img_b64: str, context: str, shortlist_sci=None, model: str = None):
    """Species ID from the cropped photo, GROUNDED on the on-device model's closest matches. Pure free-ID
    (no shortlist) was measured to hallucinate a confident random local species on blurry/ambiguous photos
    (a Costa Rica ani → Long-billed Hermit / barbet / finch, inventing field marks); anchoring on ONE guess
    instead made the model rubber-stamp the common look-alike. The middle path — the on-device top-K as
    grounding, override only with cited visible marks, no defaulting to the locally-common species, and a low
    confidence when the image is unclear — keeps clear photos sharp while staying in the right family on hard
    ones. We reconcile the pick with the on-device model in code afterwards."""
    prompt = _IDENTIFY_PROMPT.format(
        context=(context + "\n") if context else "",
        shortlist=("\n".join(f"- {s}" for s in shortlist_sci) if shortlist_sci else "- (none available)"))
    body = json.dumps({
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0.15, "maxOutputTokens": 8000,
                             "thinkingConfig": {"thinkingBudget": -1}},
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model or GEMINI_ESCALATE_MODEL}:generateContent?key={GEMINI_API_KEY}"
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        resp = json.load(r)
    txt = ""
    for p in resp.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        if "text" in p:
            txt += p["text"]
    lines = txt.splitlines()
    ordered = [l for l in lines if l.strip().upper().startswith("FINAL")] + lines
    for ln in ordered:
        cand = ln.strip()
        if cand.upper().startswith("FINAL"):
            cand = cand[5:].lstrip(": ").strip()
        m = _ONE.match(cand)
        if m:
            return {"common": _demd(m.group(1)), "sci": _demd(m.group(2)),
                    "confidence": int(m.group(3)), "reason": m.group(4).strip(), "raw": txt.strip()}
    return {"raw": txt.strip()}


_AGENTIC_PROMPT = (
    "You are an expert field ornithologist. Identify the bird in this photo to a single species.\n"
    "{context}"
    "A bird-recognition model's closest visual matches were:\n{shortlist}\n"
    "The true bird is most likely one of these or a close relative.\n"
    "The photo may be blurry — USE CODE EXECUTION to crop/zoom into the bird's HEAD and BILL and contrast-"
    "stretch it so you can actually read the bill shape and eye, then look again.\n"
    "Reason: 1) BILL depth/curvature (deep+arched/heavy vs slender+pointed), tail-to-body proportion, posture "
    "— note what is CLEAR vs masked by blur. 2) candidate families from the silhouette geometry. 3) the single "
    "best-fit species AT THIS LOCATION by structure — do not default to the commonest species or state a mark "
    "you cannot see.\n"
    "If still unclear, give a best guess but LOW confidence (<=55).\n"
    "Finish with a line, alone, EXACTLY: FINAL: Common Name (Scientific name) - NN% - reason"
)


def _gemini_identify_agentic(img_b64: str, context: str, shortlist_sci=None):
    """Aggressive re-check for UNCERTAIN birds: gemini-3.6-flash with the code_execution tool (agentic
    vision). It writes Python to zoom/contrast-stretch the head+bill crop and reasons structurally, pulling
    signal a single forward pass misses (measured: cracked a blurry Groove-billed Ani that the single pass
    called a hummingbird). ~3x the tokens of the plain pass (the sandbox re-ingests the image), so it fires
    only on the hard minority (see IDENTIFY_UNSURE_CONF)."""
    prompt = _AGENTIC_PROMPT.format(
        context=(context + "\n") if context else "",
        shortlist=("\n".join(f"- {s}" for s in shortlist_sci) if shortlist_sci else "- (none available)"))
    body = json.dumps({
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
            {"text": prompt},
        ]}],
        "tools": [{"code_execution": {}}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8000,
                             "thinkingConfig": {"thinkingBudget": -1}},
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_ESCALATE_MODEL}:generateContent?key={GEMINI_API_KEY}"
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=150) as r:
        resp = json.load(r)
    txt = ""
    for p in resp.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        if "text" in p:
            txt += p["text"]
    lines = txt.splitlines()
    ordered = [l for l in lines if l.strip().upper().startswith("FINAL")] + lines
    for ln in ordered:
        cand = ln.strip()
        if cand.upper().startswith("FINAL"):
            cand = cand[5:].lstrip(": ").strip()
        m = _ONE.match(cand)
        if m:
            return {"common": _demd(m.group(1)), "sci": _demd(m.group(2)),
                    "confidence": int(m.group(3)), "reason": m.group(4).strip(), "raw": txt.strip()}
    return {"raw": txt.strip()}


def _identify_unsure(gem) -> bool:
    """Fire the agentic re-check when the fast grounded pass is unsure — no valid pick, or low confidence.
    (Low confidence is the honest 'hard bird' signal: robins/grackles come back 95-100 and skip it.)"""
    if not gem or not gem.get("sci"):
        return True
    return (gem.get("confidence") or 0) < IDENTIFY_UNSURE_CONF


def _consensus_from_gem(gem, sl_lc, od_top, od, latf, lngf, region):
    """Reconcile a Gemini verdict into a consensus dict (agree / gemini / gemini_offcatalog), range-gated,
    falling back to the on-device best in-range pick. Shared by the sync panel and the async agentic worker."""
    if gem and gem.get("sci"):
        gsci_lc = gem["sci"].lower()
        canon = _LABEL_LC.get(gsci_lc)
        g_in_range = bool(_range_ok(np.array([RANGE_SCI2ROW.get(canon, -1)], np.int64), latf, lngf)[0]) if canon else True
        if canon and (gsci_lc in sl_lc or g_in_range or gem.get("confidence", 0) >= 85):
            src = "agree" if (od_top and gsci_lc == od_top.lower()) else "gemini"
            return {"sci": canon, "common": gem.get("common"), "confidence": gem.get("confidence", 80),
                    "source": src, "reason": gem.get("reason")}
        if not canon and gem.get("confidence", 0) >= 75:
            print("OFFCATALOG " + json.dumps({"sci": gem["sci"], "common": gem.get("common"),
                                               "region": region, "confidence": gem.get("confidence")}), flush=True)
            return {"sci": gem["sci"], "common": gem.get("common"), "confidence": gem.get("confidence"),
                    "source": "gemini_offcatalog", "reason": gem.get("reason"), "in_catalog": False}
    return {"sci": od_top, "common": (od[0].get("common") if od else None),
            "confidence": int((od[0].get("confidence", 0) or 0) * 100) if od else None,
            "source": "classifier", "reason": None}


# Async agentic re-check: the code_execution pass is ~32s (exceeds the gateway timeout), so /panel returns
# the fast grounded verdict immediately + a task id and runs the deep look in a background thread; the app
# polls GET /agentic/{id}. In-memory store is fine — single uvicorn container, and the poll hits it.
AGENTIC_ASYNC = os.environ.get("AGENTIC_ASYNC", "1") == "1"
_AGENTIC_TASKS = {}          # id -> {"status": "pending"|"done"|"error", "consensus": {...}|None, "ts": float}
_AGENTIC_LOCK = threading.Lock()
_AGENTIC_TTL = 300.0         # forget finished tasks after 5 min


def _agentic_worker(task_id, crop_b64, ctx_str, candidates, sl_lc, od_top, od, latf, lngf, region):
    try:
        ag = _gemini_identify_agentic(crop_b64, ctx_str, candidates)
        cons = _consensus_from_gem(ag, sl_lc, od_top, od, latf, lngf, region) if (ag and ag.get("sci")) else None
        st = {"status": "done", "consensus": cons, "ts": time.time()}
        print("AGENTIC-DONE " + json.dumps({"task": task_id[:8], "consensus": cons}, default=str), flush=True)
    except Exception as ex:
        st = {"status": "error", "consensus": None, "ts": time.time()}
        print("AGENTIC-ERR " + repr(ex), flush=True)
    with _AGENTIC_LOCK:
        _AGENTIC_TASKS[task_id] = st


def _panel_unsure(gem, shortlist) -> bool:
    """Should we escalate the cheap flash-lite judge to the strong (2.5-flash + reasoning) one?
    Escalate on any sign of a hard/contested call: the light model wants a species off the shortlist,
    disagrees with the classifiers' #1 pick, gave a low confidence, or didn't parse. (Blind spot, by
    design: when the phone + BioCLIP + flash-lite all confidently AGREE on the same wrong common bird,
    nothing looks unsure and we don't escalate — that class needs a stronger base model, not a gate.)"""
    if not gem or not gem.get("sci"):
        return True
    sl = [s.lower() for s in shortlist]
    g = gem["sci"].lower()
    if g not in sl:
        return True                                   # names something off the shortlist
    if sl and g != sl[0]:
        return True                                   # disagrees with the classifiers' top pick
    if (gem.get("confidence") or 0) < 80:
        return True                                   # not confident
    return False


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
    oor = set()  # scis clearly out-of-range here (GBIF says the species doesn't occur at this location)
    if latf is not None and lngf is not None and od:
        od_ok = _range_ok(np.array([RANGE_SCI2ROW.get(str(d["sci"]), -1) for d in od], np.int64), latf, lngf)
        for d, okv in zip(od, od_ok):
            if not okv:
                d["confidence"] = float(d.get("confidence", 0) or 0) * RANGE_OUT_FACTOR
                oor.add(str(d["sci"]).lower())
        od.sort(key=lambda d: -float(d.get("confidence", 0) or 0))
    od_scis = [str(d["sci"]) for d in od]
    # Shortlist handed to the fast-path Gemini: drop species clearly out-of-range here, so Gemini can't
    # lock onto an out-of-range look-alike on a blurry photo (e.g. African barbet at a BC location).
    # Falls back to the full list if that would empty it (no location / all in-range).
    od_scis_ir = [s for s in od_scis if s.lower() not in oor] or od_scis

    # Location/date context, shared by the fast path and the full-panel adjudication.
    _ctx = []
    if region.strip():
        _ctx.append(f"The photo was taken in/near {region.strip()}.")
    elif lat.strip() and lng.strip():
        _ctx.append(f"The photo was taken near {lat.strip()}, {lng.strip()}.")
    if date.strip():
        _ctx.append(f"Date: {date.strip()}.")
    ctx_str = " ".join(_ctx)

    # GEMINI-FIRST, on-device + Gemini only (BioCLIP retired — it misfired on busy real-world scenes,
    # e.g. a Fijian sparrowhawk as top pick for a Costa Rica photo). We do NOT hand Gemini the phone's
    # guess: naming it anchors even the strong model onto the common look-alike (measured: it flips a
    # Groove-billed Ani to Great-tailed Grackle every run). Gemini IDs the cropped bird cleanly, then we
    # reconcile with the on-device model in code. (Mike, 2026-07-28.)
    # Cost gate (Mike, 2026-07-28): a confident on-device pick is confirmed by the CHEAP model — it nails
    # unmistakable birds and just agrees; only uncertain birds (the hard look-alikes flash-lite gets
    # confidently wrong) escalate to the strong model. Out-of-range picks are already conf-demoted upstream,
    # so a suspect pick falls below the threshold and escalates too.
    od_top_conf = float(od[0].get("confidence", 0) or 0) if od else 0.0
    id_model = GEMINI_MODEL if od_top_conf >= PANEL_CONFIDENT_CONF else GEMINI_ESCALATE_MODEL
    crop_b64 = _crop_b64(img)
    gem = None
    if GEMINI_API_KEY:
        try:
            gem = _gemini_identify(crop_b64, ctx_str, od_scis_ir[:6], model=id_model)   # grounded on on-device top-K
        except Exception:
            gem = None

    od_top = od_scis_ir[0] if od_scis_ir else (od_scis[0] if od_scis else None)
    shortlist = (od_scis_ir[:5] or od_scis[:5])
    sl_lc = {s.lower() for s in shortlist}
    consensus = _consensus_from_gem(gem, sl_lc, od_top, od, latf, lngf, region)

    # ASYNC agentic deep-look: on an UNSURE fast result, kick off the ~32s code-execution re-check in a
    # background thread and hand the app a task id to poll (GET /agentic/{id}). The fast verdict returns NOW,
    # so no gateway timeout; the card refreshes if the deep look shifts the ID (e.g. grackle → ani).
    agentic_task = None
    if AGENTIC_ASYNC and GEMINI_API_KEY and _identify_unsure(gem):
        agentic_task = uuid.uuid4().hex
        with _AGENTIC_LOCK:
            now = time.time()
            for _k in [k for k, v in _AGENTIC_TASKS.items() if now - v.get("ts", now) > _AGENTIC_TTL]:
                _AGENTIC_TASKS.pop(_k, None)
            _AGENTIC_TASKS[agentic_task] = {"status": "pending", "consensus": None, "ts": now}
        threading.Thread(target=_agentic_worker, daemon=True,
                         args=(agentic_task, crop_b64, ctx_str, od_scis_ir[:6],
                               sl_lc, od_top, od, latf, lngf, region)).start()

    result = {"consensus": consensus, "shortlist": shortlist,
              "has_agentic_pending": bool(agentic_task), "agentic_task_id": agentic_task,
              "judges": {"ondevice": od[:8], "bioclip": None, "merlin": None, "gemini": gem}}
    print("PANEL " + json.dumps({"consensus": consensus, "region": region, "date": date,
                                 "ondevice_top": od_scis[:1], "od_conf": round(od_top_conf, 3),
                                 "id_model": id_model, "agentic_pending": bool(agentic_task),
                                 "gemini": (gem or {}).get("sci")}, default=str), flush=True)
    return result


@app.get("/agentic/{task_id}")
async def agentic_status(task_id: str, authorization: str = Header(default="")):
    """Poll the async agentic deep-look. `pending` = still running (~32s); `done` returns the refined
    consensus (may equal the fast one, or shift the ID); `unknown` = expired/never existed."""
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")
    with _AGENTIC_LOCK:
        t = _AGENTIC_TASKS.get(task_id)
    if not t:
        return {"status": "unknown"}
    return {"status": t["status"], "consensus": t.get("consensus")}
