"""Assistive Vision — FastAPI server (hardened).

POST /analyze   multipart: frame=<jpeg>, keyword=<optional str>
GET  /          mobile web app
GET  /health    {ready, warming, object_model, text_obb}

Run:  python -m server.main            (or)
      uvicorn server.main:app --host 0.0.0.0 --port 8000
"""
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from . import (classes_av, config, detector, priority, spatial, symbols,
               text_pipeline)

VERSION = "1.7"
WEB = Path(__file__).resolve().parent.parent / "web" / "index.html"

# ML models are not thread-safe; serialize inference.
_infer_lock = threading.Lock()

# /analyze is a sync def, so FastAPI runs it in a threadpool worker — several
# concurrently. _sessions is mutated from all of them; without this its
# eviction pass could raise "dictionary changed size during iteration".
# Always taken INSIDE _infer_lock, never the reverse, so the two cannot deadlock.
_sessions_lock = threading.Lock()
_state = {"ready": False, "warming": False, "error": None}

MAX_SIDE = 960  # resize cap: bounds CPU latency, plenty for signboards

# A 960px JPEG at q0.8 is ~150 KB; 8 MB leaves room for an uncompressed
# full-resolution phone capture while bounding what one request can allocate.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_SESSION_LEN = 64        # session ids are client-supplied dict keys
MAX_SESSIONS = 500
SESSION_TTL_S = 120

# How long /analyze will wait for the inference lock before shedding. The web
# client aborts at 25 s and a worst-case OCR frame costs ~7 s (runs/eval), so
# waiting longer than this can only produce work nobody is still listening for.
INFER_WAIT_S = 10.0

# /analyze burns 1-3 s of CPU per call and needs no credentials, so an
# unthrottled one is free compute for anyone who finds the URL. A live session
# sends a frame every 2.6 s (~23/min); 60 leaves headroom for scans on top.
RATE_LIMIT = 60             # requests per window, per client IP
RATE_WINDOW_S = 60
MAX_TRACKED_IPS = 2000

_rate: dict = {}
_rate_lock = threading.Lock()


# ---------------------------------------------------------------- warmup
def _warmup():
    _state["warming"] = True
    try:
        detector._get_model()          # downloads yolov8n.pt on first run
        text_pipeline._get_reader()    # downloads EasyOCR en models
        text_pipeline._get_obb()       # optional fine-tuned text model
        _state["ready"] = True
    except Exception as e:  # pragma: no cover
        _state["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        _state["warming"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(title="Assistive Vision API", version=VERSION, lifespan=lifespan)


# ---------------------------------------------------------------- misc
@app.middleware("http")
async def limit_body(request: Request, call_next):
    """Reject oversized uploads before Starlette parses/spools the body.

    /analyze read the whole multipart body into memory with no cap, so a
    single large POST could exhaust RAM on the 512 MB-class free tiers this
    is deployed to.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"error": f"Frame too large (limit {MAX_UPLOAD_BYTES // 1024} KB)."},
            status_code=413)
    return await call_next(request)


def _rate_limited(ip: str) -> bool:
    """Sliding-window count of recent hits from one IP."""
    now = time.time()
    with _rate_lock:
        hits = [t for t in _rate.get(ip, ()) if now - t < RATE_WINDOW_S]
        _rate[ip] = hits
        if len(hits) >= RATE_LIMIT:
            return True
        hits.append(now)
        # IPs are attacker-supplied keys, so the table needs its own bound
        if len(_rate) > MAX_TRACKED_IPS:
            for k in [k for k, v in list(_rate.items())
                      if not v or now - v[-1] >= RATE_WINDOW_S]:
                _rate.pop(k, None)
        return False


# Added last, so it is the OUTERMOST middleware: the cheap check runs before
# anything parses a body.
@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Throttle the expensive endpoint per client IP.

    The wildcard CORS headers that used to live here are gone: the web app is
    served by this same app (`API = location.origin`), so it is same-origin and
    never needed them — they only let third-party sites spend our CPU.
    """
    if request.url.path == "/analyze":
        # ponytail: X-Forwarded-For is spoofable when not behind a proxy. This
        # deploys behind HF Spaces / cloudflared, which both set it; keying on
        # request.client.host there would bucket every user into one counter.
        # Move to a proxy/edge-level limit if the hosting changes.
        fwd = request.headers.get("x-forwarded-for", "")
        ip = fwd.split(",")[0].strip() or (
            request.client.host if request.client else "unknown")
        if _rate_limited(ip):
            return JSONResponse(
                {"error": "Too many requests. Slow down."},
                status_code=429, headers={"Retry-After": str(RATE_WINDOW_S)})
    return await call_next(request)


@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception):
    """Log the detail, return a generic message.

    This used to hand the caller f"{type(exc).__name__}: {exc}", which leaks
    internals -- exception text routinely carries filesystem paths, model paths
    and occasionally request data -- to an endpoint that takes no credentials.
    The reference is what keeps the generic message debuggable: it is printed
    beside the traceback, so a user reporting "error 3f2a1c" points straight at
    one log entry, which timestamps alone cannot do under concurrency.
    """
    ref = uuid.uuid4().hex[:6]
    print(f"[error {ref}] {request.method} {request.url.path}", flush=True)
    traceback.print_exc()
    return JSONResponse(
        {"error": f"Internal server error (reference {ref}).", "ref": ref},
        status_code=500)


@app.get("/")
def index():
    return FileResponse(WEB)


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    try:
        obb = text_pipeline._obb is not None
    except Exception:
        obb = False
    body = {
        "status": "error" if _state["error"] else "ok",
        "version": VERSION,
        "ready": _state["ready"],
        "warming": _state["warming"],
        "error": _state["error"],
        "object_model": config.OBJECT_MODEL,
        # the schema the loaded model actually speaks, not which file was found:
        # "av7" | "coco" | null (not loaded). A model whose classes match
        # neither is refused at warmup and surfaces here as an "error".
        "object_schema": detector.active_schema(),
        # Can the loaded model raise the interrupt-everything alert at all?
        # stairs_down is trained in the schema (CRITICAL_ACTIVE), but the
        # "Warning! Stop and proceed carefully" branch is only reachable when
        # the loaded weights actually speak that schema — a COCO fallback
        # cannot emit the class, and /health must not claim an alert the
        # running model cannot raise.
        "critical_alert": (detector.active_schema() == detector.SCHEMA_AV
                           and classes_av.CRITICAL_ACTIVE),
        "dormant_hazards": sorted(classes_av.DORMANT_HAZARDS),
        "text_obb": obb,
    }
    # 503 on a failed warmup. A 200 here told the orchestrator the container was
    # healthy, so a worker whose models never loaded — and which therefore 503s
    # every /analyze forever — was never restarted.
    return JSONResponse(body, status_code=503 if _state["error"] else 200)


def _pub(items):
    out = []
    for it in items:
        d = {}
        for k, v in it.items():
            if k == "box":
                d[k] = [int(x) for x in v]
            elif isinstance(v, (np.floating, np.integer)):
                d[k] = v.item()
            else:
                d[k] = v
        out.append(d)
    return out


# ---------------------------------------------------------------- analyze
# ---- temporal confirmation: an object must appear in 2 consecutive live
# frames before it is spoken (kills single-frame false positives) ----
_sessions: dict = {}


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union


def _evict_locked(now: float):
    """Bound _sessions. Caller must hold _sessions_lock.

    Both writers go through this: _ocr_due added ':ocr' keys without ever
    pruning, so the table could sit above MAX_SESSIONS between _confirm calls.
    """
    if len(_sessions) <= MAX_SESSIONS:
        return
    for k in [k for k, v in list(_sessions.items())
              if now - v["ts"] > SESSION_TTL_S]:
        _sessions.pop(k, None)
    # age alone cannot bound a burst of fresh client-supplied keys —
    # they are all young. Evict oldest-first down to the cap.
    if len(_sessions) > MAX_SESSIONS:
        oldest = sorted(_sessions.items(), key=lambda kv: kv[1]["ts"])
        for k, _ in oldest[:len(_sessions) - MAX_SESSIONS]:
            _sessions.pop(k, None)


def _confirm(session: str, items):
    """Confirmed = same label AND overlapping position in the previous frame.

    Label alone was not enough: a person on the far left of frame 1 and a
    different person on the far right of frame 2 both counted as "confirmed",
    so the position check the confirmation exists for never happened.
    """
    now = time.time()
    with _sessions_lock:
        prev = _sessions.get(session, {}).get("items", [])
        _sessions[session] = {"items": [(i["label"], i["box"]) for i in items],
                              "ts": now}
        _evict_locked(now)
    for i in items:
        i["confirmed"] = any(lbl == i["label"] and
                             _iou(box, i["box"]) >= config.CONFIRM_IOU
                             for lbl, box in prev)
    return [i for i in items if i["confirmed"]]


def _ocr_due(gate: str) -> bool:
    """Live-mode OCR gate. EasyOCR is the dominant per-frame cost, so between
    keyword searches we run it every OCR_EVERY_N frames and reuse the previous
    result; object/hazard detection still runs on every frame."""
    now = time.time()
    with _sessions_lock:
        st = _sessions.setdefault(gate, {"n": 0, "texts": [], "ts": now})
        st["n"] += 1
        st["ts"] = now
        _evict_locked(now)
        return (st["n"] - 1) % config.OCR_EVERY_N == 0


# NOTE: sync def on purpose — FastAPI runs it in a worker thread, so the
# blocking ML inference never freezes the event loop / health endpoint.
@app.post("/analyze")
def analyze(frame: UploadFile = File(...), keyword: str = Form(""),
            mode: str = Form("single"), session: str = Form(""),
            pitch: float = Form(0.0), vfov: float = Form(0.0)):
    if not _state["ready"]:
        msg = ("AI models failed to load: " + _state["error"]) if _state["error"] \
            else "AI models are still loading on the server. Please wait."
        return JSONResponse({"error": msg, "warming": _state["warming"]},
                            status_code=503)

    t0 = time.time()
    # capped read: bounds our allocation even if the body arrived without a
    # Content-Length header for limit_body to check
    raw = frame.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"error": f"Frame too large (limit {MAX_UPLOAD_BYTES // 1024} KB)."},
            status_code=413)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Could not decode image."}, status_code=400)
    session = session[:MAX_SESSION_LEN]

    # bound latency: downscale huge frames
    h0, w0 = img.shape[:2]
    scale = MAX_SIDE / max(h0, w0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)),
                         interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    # ---- blur gate: motion-blurred frames produce garbage detections ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < config.BLUR_THRESHOLD:
        return {"speech": "Hold the camera steady.", "blur": True,
                "hazard_count": 0, "keyword": "", "match": None,
                "hazards": [], "objects": [], "texts": [], "symbols": [],
                "frame": {"w": w, "h": h},
                "ms": round((time.time() - t0) * 1000)}

    # a keyword search needs fresh text to find its match; only the idle live
    # loop is gated
    gate = session + ":ocr"
    live_idle = mode == "live" and bool(session) and not keyword.strip()
    run_ocr = (not live_idle) or _ocr_due(gate)

    # Shed load rather than queue it. Every request blocked here holds a
    # threadpool worker, and the client gives up at 25 s (index.html), so a
    # deep queue means we finish inference for callers who already left --
    # spending the CPU that made the queue deep in the first place. Failing
    # fast at INFER_WAIT_S leaves the worst case (wait, then a ~7 s OCR frame)
    # inside the client's budget, and a 503 is something it can retry.
    if not _infer_lock.acquire(timeout=INFER_WAIT_S):
        return JSONResponse(
            {"error": "Server busy, try again in a moment.", "busy": True},
            status_code=503, headers={"Retry-After": "2"})
    try:  # models are not thread-safe
        objects, hazards = detector.detect_objects(img)
        if run_ocr:
            texts = text_pipeline.detect_text(img)
        else:
            with _sessions_lock:
                texts = list(_sessions.get(gate, {}).get("texts", []))
    finally:
        _infer_lock.release()
    if live_idle and run_ocr:
        with _sessions_lock:
            st = _sessions.get(gate)
            if st is not None:      # a concurrent _confirm may have evicted it
                st["texts"] = texts

    pitch = max(-10.0, min(70.0, pitch))
    vf = vfov if 25.0 <= vfov <= 90.0 else None
    for group in (objects, hazards, texts):
        spatial.annotate(group, w, h, pitch, vf)

    # hazards only interrupt when close — distant ones demote to objects
    close_hz = [x for x in hazards if x["proximity"] != "at a distance"]
    objects += [{**x, "kind": "object"} for x in hazards if x not in close_hz]

    syms = symbols.symbols_from_objects(objects + close_hz) \
        + symbols.symbols_from_texts(texts)

    kw = symbols.clean_query(keyword)
    match = symbols.match_keyword(kw, texts, syms, objects) if kw else None

    # live mode: only CONFIRMED (2 consecutive frames) objects/hazards are
    # spoken; everything is still returned for rendering
    if mode == "live" and session:
        speak_hz = _confirm(session + ":h", close_hz)
        speak_obj = _confirm(session + ":o", objects)
    else:
        speak_hz, speak_obj = close_hz, objects

    out = priority.build_speech(speak_hz, speak_obj, texts, syms, kw or None, match)

    return {
        "speech": out["speech"],
        "hazard_count": out["hazard_count"],
        "blur": False,
        "keyword": kw,
        "match": _pub([match])[0] if match else None,
        "hazards": _pub(close_hz),
        "objects": _pub(objects),
        "texts": _pub(texts),
        "symbols": _pub(syms),
        "frame": {"w": w, "h": h},
        "ms": round((time.time() - t0) * 1000),
    }


if __name__ == "__main__":  # python -m server.main
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000)
