# Assistive Vision System for Visually Impaired Users Using Oriented Text Detection

## Project overview
A mobile web visual-assistance prototype backed by local FastAPI inference. It reads
signs using oriented text regions and EasyOCR, detects supported objects with YOLO,
and speaks concise, prioritized information. It is an assistive research prototype,
not a replacement for mobility aids, judgment, or a trained mobility professional.

## Features and available modes
- **Live monitoring:** say Start, Start scan or Start scanning to start camera and both
  inference lanes. Announce stable new detections or meaningful changes once.
- **Environment scan:** say Scan environment for one concise current-view summary,
  then automatically return to change-based monitoring.
- **Keyword Search:** continuous scanning, normalized whole-token phrases, restricted
  one-edit fuzzy matching on long words, exact digits, two-frame text confirmation.
- **Navigation:** browser geolocation and an automatic Google Maps directions handoff after the spoken prompt.
  Distance and time reuses the selected destination. Maps resolves the destination
  and supplies route metrics; no local route distance is invented.
- Large Start/mode/Stop/Repeat buttons, keyboard focus, zoom, high contrast, text
  transcript, voice commands, and portrait camera capture.
- Priority speech queue with cooldown, bounded backlog and hazard interruption.
- Partial inference failures are disclosed while available components continue.
- No permanent camera-frame or microphone recording by the application.

## Architecture and system flow
```text
Phone camera -> resized JPEG -> FastAPI -> object detector + oriented text/OCR
                                         -> spatial analysis -> temporal checks
                                         -> search/context/priority -> JSON
Phone <- polygons + warnings + speech text <- response
Phone speech queue -> browser TTS
Navigation: browser GPS -> local handoff provider -> user opens Google Maps
```
`server/interfaces.py` defines replaceable vision/navigation contracts and local
adapters. The existing `startCamera`/`capture` functions form the browser camera
boundary; a wearable client can post JPEG frames to the same API. Speech recognition
and synthesis use browser APIs, so the server does not need microphone access.

## Technology stack and repository layout
Python 3.11/3.12, FastAPI, Pydantic, OpenCV, Ultralytics YOLO, EasyOCR (CRAFT),
vanilla HTML/CSS/JavaScript, browser media/geolocation/speech APIs.

| Location | Responsibility |
|---|---|
| `server/main.py` | API, bounded uploads, model warmup, independent object/OCR locks |
| `server/config.py`, `.env.example` | Validated runtime options |
| `server/detector.py`, `server/text_pipeline.py` | Model adapters, oriented quads and rectified OCR |
| `server/spatial.py` | Directions and conservative approximate object steps |
| `server/symbols.py`, `server/environment.py`, `server/priority.py` | Matching, summary selection, hazard ordering |
| `server/navigation.py` | Validated navigation request and Maps fallback |
| `web/index.html`, `web/app.js` | Existing accessible UI and one orchestration entry point |
| `web/workflow.js`, `web/speech.js`, `web/realtime.js` | Command parser, singleton speech lifecycle, tracking and latest-frame lanes |
| `scripts/` | Dataset converters, validation/splitting, training, evaluation, HTTPS launcher |
| `tests/` | Python regressions and JavaScript state/queue tests |
| `docs/IMPLEMENTATION_LOG.md` | Inspection, changes, verification and limitations |

## Installation and environment setup
Run from the project root. Existing workspace already contains `.venv` and cached
models; a fresh installation needs dependencies and model downloads.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```
On Linux/macOS use `python3 -m venv .venv` and `.venv/bin/python` instead.
For runtime-only installations, use `requirements.txt`. Use Node 24 for frontend
tests; no frontend package installation or build is needed.

## Running backend and frontend
```powershell
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 127.0.0.1 --port 8000 --env-file .env
```
Open **http://localhost:8000**. FastAPI serves the frontend at the same origin.
Interactive API documentation is at **http://localhost:8000/docs**.
`python -m server.main` also runs the app but does not load `.env` automatically.

Environment options include model paths, CPU/GPU device, inference size, confidence,
NMS, frame interval (milliseconds), OCR interval (sampled frames), speech cooldown
(seconds), step length, camera height/FOV, language and TTS rate. Do not put secrets
in the frontend. `.env` and `certs/` are ignored by Git.

## Model setup
Default object model: `yolov8n.pt` (COCO), downloaded by Ultralytics if missing.
Optional evaluated AV-6 model: `models/av_obstacle.pt`. Schema is validated from the
model class names. Two candidate checkpoints exist in this workspace; they have not
been promoted by this implementation. The AV-6 vocabulary lacks vehicles and must
not be represented as an outdoor hazard detector.

Default text path: cached or first-run downloaded EasyOCR English CRAFT detector and
recognizer. Output retains all four quad corners and an angle. Optional custom
`models/text_obb.pt` must be an OBB model; its proposals are perspective-rectified
before OCR. Invalid optional OBB weights fall back to CRAFT and warmup health reports
the unavailable component. OCR confidence gates recognized text.

No text OBB checkpoint is present. **MODEL TRAINING NOT YET COMPLETED** for the full
requested custom text/hazard coverage. Historical candidate runs are retained;
no new training or accuracy claims are made here. Pits and descending stairs are
not supported by the default model. Review model/dataset licenses separately from
the repository's MIT license before distributing weights.

## HTTPS smartphone testing
Use a trusted HTTPS origin for phone camera/microphone access. A phone's localhost
is the phone itself; use the computer's LAN address on the same trusted network.
No LAN address is hard-coded.

With [mkcert](https://github.com/FiloSottile/mkcert) installed, create a local
certificate containing your chosen LAN address:
```powershell
$lanAddress = Read-Host 'Computer LAN IP address'
New-Item -ItemType Directory -Force certs
mkcert -install
mkcert -cert-file certs/dev.pem -key-file certs/dev-key.pem localhost 127.0.0.1 $lanAddress
.\scripts\run_https.ps1 -Certificate certs/dev.pem -Key certs/dev-key.pem
```
Follow mkcert's mobile instructions to install/trust **rootCA.pem** on your own test
phone. Never share **rootCA-key.pem**. Open `https://<LAN-address>:8443` on the phone.
The certificate must cover that address. Allow the server port on your private LAN
if needed. Certificate installation and phone testing are manual setup steps;
they have not been performed on your phone in this session.

The launcher uses Uvicorn's documented [TLS and environment-file settings](https://www.uvicorn.org/settings/).
A trusted HTTPS tunnel is another option, but it exposes the server and sends frames
through the tunnel provider. This prototype has no user authentication: keep it on
a trusted local network unless you add authenticated access and edge rate limits.

## Voice commands and controls
On opening, the app attempts the welcome prompt and microphone activation. Say
**Start scanning**, **Start scan**, **Start** or **Begin scanning** to enter live
monitoring. A browser may require a first tap or microphone permission; Start and
Enable voice remain available. Browser security restrictions cannot be bypassed.

| Command | Action |
|---|---|
| Scan environment / describe environment / what is around me | One current-view summary, then monitoring |
| Find washroom / pharmacy / exit / emergency exit / room 101 / stairs / reception | Persistent visual search |
| Navigate to / go to / take me to a destination | GPS with a deadline, then automatic Maps directions |
| Distance and time / how far is it / travel time | Maps directions for the remembered destination |
| Stop / stop scanning / stop camera / stop microphone | Stop camera, recognition and inference requests |
| Back / go back / main menu | Return to the assistance menu |
| Repeat / help / mute / unmute | Speech controls |

One SpeechRecognition instance is reused. Recognition is stopped and its `onend`
event is awaited before TTS; results are rejected during speech and a configurable
post-speech cooldown. A late old utterance cannot restart listening. The speech
queue is bounded and obsolete normal messages are discarded. If the recognizer
fails to stop, it is disabled rather than starting another session.

Navigation uses the same tab to avoid popup blocking. If GPS is denied or times out,
Maps receives the destination without a fabricated origin. Network/app handoff may
still fail outside this page; the visible Maps link remains a manual retry until
navigation leaves the page. Route distance/time are provided by Google Maps.

Stop and page backgrounding cancel both processing lanes, stop media tracks and
clear overlays. Voice can be re-enabled with the persistent Enable voice control.
Calibration accepts an optional measured vertical field of view; distances remain
approximate. Ordinary scanning never asks the user to steer or steady the camera.

## API documentation
All image routes accept multipart `frame`, optional `keyword` (max 200 characters),
`mode` (`single` or `live`), `session`, `pitch` and `vfov`. An optional `component` selects `all`, `objects` or `text`; the two detection
paths select their named component automatically. Live clients should use a
fresh session ID per camera session. Frames are limited to 8 MiB including multipart
body and 20 million decoded pixels; inference resizes to at most 960 pixels per side.

| Endpoint | Behavior |
|---|---|
| `GET /api/health` or `/health` | Model readiness, actual schema, degraded components |
| `GET /api/config` | Public client options only |
| `POST /api/analyze/frame` or `/analyze` | Integrated frame processing |
| `POST /api/search`, `/api/environment` | Same integrated pipeline; search supplies keyword |
| `POST /api/detect/text`, `/api/detect/objects` | Independent OCR and object inference; the browser runs one request per lane |
| `POST /api/navigation` | JSON destination, optional latitude/longitude pair -> honest Maps handoff |

Responses include `texts` (quads/confidence), objects, hazards, match, speech, priority,
component errors, dimensions and measured processing milliseconds. 400: invalid
image; 413: oversized; 422: invalid parameters; 429: rate limit; 503: loading/busy.
A component failure can return 200 with explicit `component_errors`; clients must
surface this and never interpret an empty list as a safe path.

Coordinates refer to the returned resized frame. Object and OCR lanes have separate
locks and independent browser request loops. Each loop captures the latest frame
only when ready; there is no frame backlog. A busy split endpoint returns 503
immediately. Stale responses are ignored; overlays clear on failed/expired frames.

Detection memory uses label plus normalized overlap/position with one-to-one matching.
Normal detections require persistence; high-confidence immediate hazards can bypass
it. Unchanged active tracks stay silent, missed tracks disappear from the UI, and
tracks are retired after configured misses or expiry. Reappearing retired tracks
can be announced again. This is heuristic tracking, not guaranteed identity.

Blur is quality metadata, not a stop condition. OCR uses conditional contrast
adjustment, oriented crops, bounded recognition retries and garbage filtering.
Darkness, severe blur, occlusion and missing model classes can still prevent detection.

Defaults: object request delay 350 ms, OCR delay 2000 ms, microphone cooldown 650 ms,
track lifetime 6500 ms, three missed observations, maximum accepted response age
4500 ms, GPS deadline 8000 ms, and two inference CPU threads. These are configurable
in `.env.example`. Delays are between completed requests; they are not an FPS claim.
The legacy combined API retains its frame-count OCR gate for compatibility.

## Dataset setup and training
Existing datasets, annotations, splits and candidate runs were preserved. Do not
mix COCO and AV class indices or evaluate against unreviewed pseudo-labels as truth.
Consult `docs/TRAINING.md`, `docs/DATA_SOURCING.md`, `scripts/train_obb.md`, and
`docs/ANNOTATION_BRIEF.md` for the existing workflows. Review historical counts and
claims against the actual files before using them in a paper.

Converters: `convert_icdar_to_obb.py`, `convert_cocotext_to_obb.py`,
`convert_synthtext_to_obb.py`. Split by source video using `prepare_split.py` to avoid
near-duplicate leakage. `train_local_baseline.py` and the Colab trainers enforce
class-coverage gates and write candidate checkpoints. Run each with `--help` before
choosing dataset/output paths. No dataset downloads are required or performed here.

## Testing
```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
node --test --test-isolation=none tests/test_*.cjs
.\.venv\Scripts\python.exe scripts/smoke_test.py
```
The smoke test runs real models on generated horizontal/tilted signs and saves
actual outputs to `runs/prototype-smoke.json`. This is integration verification,
not a representative OCR accuracy benchmark. Python tests use test doubles where
needed and do not prove live-camera behavior. Windows sandbox users may need to run
pytest in a normal terminal so its temporary test directories are writable.

## Evaluation
`evaluate.py` retains separate raw-model and deployed-pipeline evaluation, latency
statistics and self-evaluation safeguards. Use its `--help` and existing dataset
configuration. YOLO validation supplies model mAP; do not substitute fixed-threshold
F1 for mAP.

`evaluate_text.py` evaluates supplied four-point polygons with rotated IoU, one-to-one
matching, precision, recall, F1, matched OCR exact accuracy, CER and WER:
```powershell
.\.venv\Scripts\python.exe scripts/evaluate_text.py annotations-and-predictions.json --output runs/text-evaluation.json
```
The file format is documented in that script. OCR metrics cover matched detections;
missed text reduces detection recall. Do not use generated test fixtures as paper
results. Real camera-to-speech, direction, step-distance and command-recognition
accuracy require labeled physical trials; no values are claimed for those here.

## Troubleshooting
- Camera denied/unavailable: use HTTPS, enable camera permission in browser settings,
  close other camera users, and choose the mode again. Navigation needs no camera.
- Microphone denied/unsupported: use buttons and typed queries; voice control is optional.
- Models loading/missing: inspect `/api/health`, server logs, paths and network access
  for first-run downloads. Once weights are cached, vision runs on the computer.
- Timeout/network failure: keep the server running and check the local address. Each inference lane has at
  most one outstanding request; object and OCR lanes can run concurrently.
- OCR errors: the system retries current frames automatically and adjusts low contrast. Missing
  detections are not evidence that the scene is clear.
- Geolocation denied: Maps can choose its own origin after handoff. No distance/time
  is fabricated. The [Maps URL integration](https://developers.google.com/maps/documentation/urls/get-started)
  uses encoded destination and optional origin; no frontend API key is required.

## Limitations and future work
Monocular object distances depend on assumed class height, camera height/FOV and
floor contact. They are approximate, conservatively rounded down, and unvalidated
for mobility safety. Sign/stair step estimates are withheld. Default pitch is zero;
portrait orientation can supply a tilt hint, but gyro availability is optional.
Calibration is approximate and must be checked against measurements.

CPU inference may take many seconds; no guaranteed real-time FPS is claimed. The
COCO fallback does not cover doors, walls, stairs or pits. Custom AV-6 coverage is
also incomplete. Future work: evaluated OBB/hazard models, metric depth calibration,
robust multi-frame tracking, authenticated deployment, offline speech recognition,
an optional route provider, user trials and a chest-camera client.

## September 12 real-time bug fixes
See [root causes, changed files and validation](docs/REALTIME_FIXES.md). No models,
datasets, or new runtime libraries were added. The previously tested v1.8 app is
extended in place; v1.9 changes the voice workflow to direct scanning and automatic
navigation as requested.
