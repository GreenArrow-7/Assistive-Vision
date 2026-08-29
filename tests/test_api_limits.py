"""API-layer tests: request limits and session bounds.

These exercise the FastAPI app without loading any ML model — the body-size
middleware runs before the endpoint, and _confirm is pure dict bookkeeping.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import threading  # noqa: E402

from server import main  # noqa: E402

client = TestClient(main.app, raise_server_exceptions=False)


def test_oversized_upload_rejected_with_413():
    """/analyze read the whole body into memory with no cap, so one large
    POST could exhaust RAM on a 512 MB free tier."""
    big = b"\xff" * (main.MAX_UPLOAD_BYTES + 1024)
    r = client.post("/analyze", files={"frame": ("frame.jpg", big, "image/jpeg")})
    assert r.status_code == 413, r.status_code
    assert "too large" in r.json()["error"].lower()


def test_normal_frame_passes_the_size_gate():
    # a realistic ~150 KB frame must not be rejected by the limiter; it stops
    # at the model-readiness check (503) instead, which is the next gate
    small = b"\xff" * (150 * 1024)
    r = client.post("/analyze", files={"frame": ("frame.jpg", small, "image/jpeg")})
    assert r.status_code != 413


def test_health_reports_schema_not_file_presence():
    body = client.get("/health").json()
    assert "object_schema" in body
    # the old field inferred "custom model" from a path existing on disk
    assert "custom_object_model" not in body


# ---------- session table bounds ----------
def test_session_table_is_bounded_against_key_flooding():
    """session is a client-supplied dict key. Age-based pruning alone could
    not bound a burst of fresh ids — they are all young."""
    main._sessions.clear()
    for i in range(main.MAX_SESSIONS * 3):
        main._confirm(f"flood-{i}", [{"label": "person", "box": (0, 0, 10, 10)}])
    assert len(main._sessions) <= main.MAX_SESSIONS, len(main._sessions)
    main._sessions.clear()


def test_confirmation_still_requires_two_consecutive_frames():
    main._sessions.clear()
    b = (100, 100, 200, 300)
    items = [{"label": "person", "box": b}, {"label": "chair", "box": b}]
    assert main._confirm("s1", items) == []          # first sighting: unconfirmed
    again = main._confirm("s1", [{"label": "person", "box": b},
                                 {"label": "chair", "box": b}])
    assert {i["label"] for i in again} == {"person", "chair"}
    main._sessions.clear()


def test_confirmation_requires_overlapping_position():
    """Label alone confirmed a DIFFERENT person on the other side of the
    frame; confirmation must also require positional overlap."""
    main._sessions.clear()
    main._confirm("s2", [{"label": "person", "box": (0, 100, 120, 400)}])
    other_side = main._confirm(
        "s2", [{"label": "person", "box": (800, 100, 920, 400)}])
    assert other_side == []
    # same spot (slightly shifted, as a walking frame gap produces) confirms
    main._confirm("s3", [{"label": "person", "box": (100, 100, 220, 400)}])
    near = main._confirm("s3", [{"label": "person", "box": (130, 110, 250, 410)}])
    assert len(near) == 1
    main._sessions.clear()


def test_ocr_gate_runs_every_nth_frame():
    """EasyOCR dominates CPU; the idle live loop must reuse cached text
    between every OCR_EVERY_N-th frame."""
    from server import config
    main._sessions.clear()
    seq = [main._ocr_due("s:ocr") for _ in range(2 * config.OCR_EVERY_N)]
    expect = [(i % config.OCR_EVERY_N) == 0 for i in range(2 * config.OCR_EVERY_N)]
    assert seq == expect, seq
    main._sessions.clear()


# ---------- per-IP rate limit ----------
FRAME = {"frame": ("frame.jpg", b"jpegbytes" * 64, "image/jpeg")}


def test_analyze_is_rate_limited_per_ip():
    """/analyze costs 1-3 s of CPU and takes no credentials, so an unthrottled
    one is free compute for anyone who finds the URL."""
    main._rate.clear()
    codes = [client.post("/analyze", files=FRAME).status_code
             for _ in range(main.RATE_LIMIT + 2)]
    assert 429 not in codes[:main.RATE_LIMIT], codes[:main.RATE_LIMIT]
    assert codes[-1] == 429, codes[-3:]
    main._rate.clear()


def test_rate_limit_keys_on_the_forwarded_client_not_the_proxy():
    """Behind HF Spaces / cloudflared every request shares the proxy's socket
    address, so keying on it would let one abuser lock out every other user."""
    main._rate.clear()
    hit = lambda ip: client.post(  # noqa: E731
        "/analyze", files=FRAME, headers={"x-forwarded-for": ip}).status_code
    for _ in range(main.RATE_LIMIT):
        hit("1.1.1.1")
    assert hit("1.1.1.1") == 429
    assert hit("2.2.2.2") != 429       # a different client is unaffected
    main._rate.clear()


def test_health_and_index_are_never_rate_limited():
    """The client polls /health every 4 s while models warm up."""
    main._rate.clear()
    for _ in range(main.RATE_LIMIT + 5):
        assert client.get("/health").status_code != 429
    main._rate.clear()


# ---------- readiness ----------
def test_health_returns_503_when_warmup_failed():
    """A 200 here told the orchestrator the container was healthy, so a worker
    whose models never loaded was never restarted."""
    assert client.get("/health").status_code == 200
    main._state["error"] = "RuntimeError: boom"
    try:
        r = client.get("/health")
        assert r.status_code == 503, r.status_code
        assert r.json()["status"] == "error"
    finally:
        main._state["error"] = None


def test_no_wildcard_cors_header():
    """The web app is served by this same app, so it is same-origin. The old
    `Access-Control-Allow-Origin: *` only let third-party sites spend our CPU."""
    headers = {k.lower() for k in client.get("/health").headers}
    assert "access-control-allow-origin" not in headers


# ---------- thread safety ----------
def test_session_table_survives_concurrent_writers():
    """analyze is a sync def, so FastAPI runs it in a threadpool — several
    requests mutate _sessions at once. Unlocked, the eviction pass raised
    "dictionary changed size during iteration"."""
    main._sessions.clear()
    errs = []

    def hammer(n):
        try:
            for i in range(200):
                main._confirm(f"t{n}-{i}",
                              [{"label": "person", "box": (0, 0, 10, 10)}])
                main._ocr_due(f"t{n}-{i}:ocr")
        except Exception as e:      # pragma: no cover - the bug under test
            errs.append(e)

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errs, errs[:3]
    assert len(main._sessions) <= main.MAX_SESSIONS, len(main._sessions)
    main._sessions.clear()


# ---------- dormant critical alert is reported, not hidden (B2) ----------
def test_health_reports_the_dormant_critical_alert():
    """AV_CRITICAL names stairs_down, which is annotated but not trained, so
    the interrupt-everything branch cannot fire. A hazard system that cannot
    raise its top alert has to say so rather than let callers assume."""
    from server import classes_av as cav
    body = client.get("/health").json()
    assert body["critical_alert"] is False
    assert "stairs_down" in body["dormant_hazards"]
    assert body["dormant_hazards"] == sorted(cav.DORMANT_HAZARDS)


def test_health_never_claims_a_critical_alert_without_a_model():
    """No weights loaded => no schema => nothing can fire, whatever the
    role tables happen to say."""
    assert main.detector.active_schema() is None
    assert client.get("/health").json()["critical_alert"] is False


# ---------- load shedding ----------
def _sharp_jpeg():
    """Noise, so the frame clears the blur gate and reaches the lock."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_busy_server_sheds_instead_of_queueing():
    """Every request blocked on the inference lock holds a threadpool worker,
    and the client aborts at 25 s -- so a deep queue finishes inference for
    callers who already left, spending the CPU that made it deep."""
    main._state["ready"] = True
    main._rate.clear()
    wait = main.INFER_WAIT_S
    main.INFER_WAIT_S = 0.2                 # keep the test quick
    main._infer_lock.acquire()              # pretend another frame is running
    try:
        r = client.post("/analyze",
                        files={"frame": ("f.jpg", _sharp_jpeg(), "image/jpeg")})
        assert r.status_code == 503, r.status_code
        assert r.json()["busy"] is True
        assert r.headers.get("Retry-After") == "2"
    finally:
        main._infer_lock.release()
        main.INFER_WAIT_S = wait
        main._state["ready"] = False
        main._rate.clear()


def test_lock_is_released_when_inference_raises():
    """A leaked lock would wedge every later request into permanent 503s."""
    main._state["ready"] = True
    main._rate.clear()
    boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom"))  # noqa: E731
    real = main.detector.detect_objects
    main.detector.detect_objects = boom
    try:
        client.post("/analyze",
                    files={"frame": ("f.jpg", _sharp_jpeg(), "image/jpeg")})
    finally:
        main.detector.detect_objects = real
        main._state["ready"] = False
        main._rate.clear()
    assert main._infer_lock.acquire(timeout=1), "inference lock was leaked"
    main._infer_lock.release()


# ---------- 500s must not leak internals ----------
def test_server_error_returns_a_reference_not_the_exception():
    """The handler used to return f'{type(exc).__name__}: {exc}' to an endpoint
    that takes no credentials; exception text carries paths and request data."""
    main._state["ready"] = True
    main._rate.clear()
    secret = "C:/secret/path/model.pt not found"
    boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(secret))  # noqa: E731
    real = main.detector.detect_objects
    main.detector.detect_objects = boom
    try:
        r = client.post("/analyze",
                        files={"frame": ("f.jpg", _sharp_jpeg(), "image/jpeg")})
    finally:
        main.detector.detect_objects = real
        main._state["ready"] = False
        main._rate.clear()
    assert r.status_code == 500, r.status_code
    body = r.text
    assert secret not in body, body
    assert "RuntimeError" not in body, body
    ref = r.json()["ref"]
    assert len(ref) == 6 and ref in r.json()["error"]
