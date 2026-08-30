# Assistive Vision — AI Navigation Assistant for the Visually Impaired

Real-time assistive system: a phone browser streams camera frames to a Python
server that runs **YOLO object/obstacle detection**, **oriented text detection +
EasyOCR**, **symbol recognition**, **spatial guidance** (left / ahead / right +
proximity), and returns a **priority-ordered voice message** — hazards first,
then the user's requested location, then signs, then an environment summary.
Outdoor mode uses **GPS + Google Maps** walking navigation.

Final Year Project — Dept. of CSE, ATMECE Mysuru (2025–26).

```
Phone (web app)                     Server (FastAPI, Python)
┌─────────────────────┐   JPEG    ┌──────────────────────────────────┐
│ Camera live capture ├──────────►│ YOLOv8  → objects + hazards      │
│ Voice input (STT)   │  keyword  │ YOLO-OBB / EasyOCR → texts       │
│ TTS speech output   │◄──────────┤ Symbol map → signs               │
│ GPS → Google Maps   │   JSON    │ Spatial → direction + proximity  │
└─────────────────────┘  +speech  │ Priority engine → spoken string  │
                                  └──────────────────────────────────┘
```

## 1. Quick start (server)

```bash
git clone https://github.com/<you>/assistive-vision.git
cd assistive-vision
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate  (Python 3.11/3.12)
pip install -r requirements.txt
python -m server.main            # or: uvicorn server.main:app --host 0.0.0.0 --port 8000
```

First run auto-downloads `yolov8n.pt` (~6 MB) and EasyOCR English models
(~100 MB). CPU works; set `OCR_GPU = True` in `server/config.py` for CUDA.

Open http://localhost:8000 — the web app is served by the same server.
Models warm up in the background; the app shows a loading overlay and
unlocks automatically when `/health` reports `ready: true`.

## 2. Run on your phone

The camera API requires **HTTPS** (or localhost). Easiest tunnel:

```bash
# option A — cloudflared (free, no signup)
cloudflared tunnel --url http://localhost:8000
# option B — ngrok
ngrok http 8000
```

Open the generated `https://…` URL in **Chrome on Android** (full support:
camera, voice input, TTS, vibration, GPS). iOS Safari: everything works except
voice *input* — use the keyword chips.

## 3. Using the app

| Control | What it does |
|---|---|
| **START LIVE ASSIST** | Continuous scan every ~2.6 s; hazards trigger vibration + red flash + "Caution…" spoken first |
| **VOICE** | Say `find washroom`, `locate exit`, or `navigate to city hospital` |
| Keyword chips | One-tap search: Exit, Washroom, Lift, Reception, Pharmacy, Cafeteria, Parking, Stairs |
| **Scan once** | Single frame analysis with full environment summary |
| **Outdoor navigation** | GPS position → Google Maps walking turn-by-turn |
| Overlay colors | 🔴 hazard · 🟡 text · 🔵 object · 🟢 symbol · white dashed = your keyword match |

Speech priority (per spec): **1) hazards → 2) keyword result → 3) symbols →
4) environment summary.** Distant "hazard-class" objects are demoted to plain
objects; only close ones interrupt.

## 4. Repository layout

```
server/
  main.py           FastAPI app — POST /analyze, serves web/
  detector.py       YOLOv8 objects + obstacle split
  text_pipeline.py  YOLO-OBB (if models/text_obb.pt exists) else EasyOCR CRAFT;
                    perspective deskew of rotated quads before OCR
  symbols.py        symbol resolution + voice-query cleaning + keyword match
  spatial.py        direction (frame thirds) + proximity (bbox area ratio)
  priority.py       hazard-first speech builder
  config.py         thresholds, hazard/object class sets, symbol keyword map
  classes_av.py     AV-7 trained schema (schema id derived from the class list)
                    + 14-name annotation vocabulary, hazard roles
web/index.html      mobile web app (camera, live loop, STT, TTS, GPS nav)
scripts/            model download, dataset build, stairs up/down re-tag queue,
                    internet data sourcing
                    (fetch_videos.py, pull_open_datasets.py — see
                    docs/DATA_SOURCING.md), YOLO-OBB text training guide
tests/              146 pytest tests: pipeline logic, API limits, dataset tooling
```

## 5. Training the YOLO-OBB text detector (optional upgrade)

Out of the box, text detection uses EasyOCR's CRAFT detector (handles rotated
quads natively). To use a true **YOLOv8-OBB** text model as described in the
report, follow `scripts/train_obb.md` (ICDAR-2015 → OBB label conversion →
`yolo obb train`). Drop the result at `models/text_obb.pt` — the pipeline
switches to it automatically (see `/health`).

## 6. Tests

```bash
pip install -r requirements-dev.txt   # test-only deps, kept out of the image
python -m pytest tests/ -q
```

## 7. API

`POST /analyze` — multipart `frame` (JPEG, max 8 MB) + `keyword` (optional string)

Returns `413` if the frame exceeds the limit, `503` while models are warming,
`400` if the JPEG cannot be decoded, `429` past 60 requests/minute per client IP
(a live session sends ~23/min, so normal use never hits it).

`GET /health` returns `503` — not `200` — when model warmup failed, so an
orchestrator restarts a worker that can never serve a request.

There is no CORS header: the web app is served by this same process, so it is
same-origin. Serving the frontend from a different host is not supported.

```json
{
  "speech": "Caution. person straight ahead, very close. washroom found: washroom is on your left, nearby.",
  "hazard_count": 1,
  "match": {"label": "washroom", "direction": "on your left", "proximity": "nearby", ...},
  "hazards": [...], "objects": [...], "texts": [...], "symbols": [...],
  "frame": {"w": 960, "h": 720}, "ms": 412
}
```

## 8. Known limitations (honest notes for the viva)

* Symbol recognition is keyword/class-mapped, not a trained icon classifier —
  a custom YOLO symbol dataset is the documented upgrade path
  (`docs/ANNOTATION_BRIEF.md`; the upload bundle is built and waiting).
* Proximity is monocular (bbox-area heuristic), not metric depth.
* Outdoor turn-by-turn is delegated to Google Maps rather than re-implemented.
* Live-assist latency is CPU-bound (~0.5–2 s/frame on a laptop CPU).

## License

MIT

## 9. Deployment

### Path A — Demo from your laptop (free, 2 minutes)
```powershell
python -m server.main
.\cloudflared.exe tunnel --url http://localhost:8000
```
Open the printed https URL on your phone. Laptop must stay on.

### Path B — Permanent live URL: Hugging Face Spaces (free CPU)
This repo includes a Dockerfile ready for HF Spaces.

**Verified 2026-08-29** by an actual `docker build` + run, not by inspection:
image builds clean (793 MB of layers, 3.24 GB on disk), `/health` returns 200
with `ready: true`, `/analyze` answers real frames end-to-end (detection, OCR,
step distance, keyword search), the web app is served, and a 9 MB upload is
rejected with 413. In-container latency 3.5–6.2 s/frame on a laptop CPU.

`/health` reports `"object_schema": "coco"` in a stock image: `.dockerignore`
excludes `*.pt`, so the COCO fallback ships and the custom schema appears only
once you place `models/av_obstacle.pt` before building.

1. Create account at https://huggingface.co → New Space → SDK: **Docker** → CPU basic (free).
2. Push this repo to the Space:
   ```bash
   git remote add hf https://huggingface.co/spaces/<user>/<space>
   git push hf main
   ```
3. First build takes ~15 min (bakes YOLO + EasyOCR models into the image).
   Your app is then live at `https://<user>-<space>.hf.space` — open it on any phone, HTTPS included, camera/voice/GPS all work.

Notes: free CPU ≈ 2–5 s per frame (set live-assist expectations accordingly);
Space sleeps after 48 h idle and wakes on first visit (~1 min).

### Why not GitHub Pages / Render free?
GitHub Pages serves static files only — it cannot run the Python/YOLO backend.
Render's free tier (512 MB RAM) OOMs loading torch + EasyOCR.
# Assistive-Vision
