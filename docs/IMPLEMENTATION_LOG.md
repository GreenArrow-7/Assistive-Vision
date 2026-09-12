# Implementation log — 10 September 2026

## Milestone 1: inspection and roadmap
Implemented: inspected server, web client, tests, model files, dataset directories,
training and evaluation tooling. Existing application is preserved and extended.
Files changed: this log.
Tests run: existing pytest suite (baseline).
Issues found: pytest temporary-directory access fails in this Windows sandbox;
unsupported sign distance estimates; missing three-mode menu; one-shot search;
OCR cache carries old camera coordinates; component failures stop all inference.
Next milestone: configuration, safe spatial/search logic, resilient analysis API;
then accessible state machine, continuous search, navigation fallback and tests.

## Existing assets and constraints
FastAPI in server/, mobile client in web/index.html, local YOLO COCO weights,
two candidate AV-6 checkpoints (not deployed), cached EasyOCR models, existing
dataset converters, training scripts and evaluation harness. No text OBB checkpoint
is present. Candidate weights are not promoted without evaluation. No new training
is performed in this implementation. MODEL TRAINING NOT YET COMPLETED for the
requested full hazard/text class coverage; historical candidate runs remain intact.

## Interface design
Keep the existing burgundy/gold camera interface. Use background #0b0508,
surface #201218, text #f6f1ee, secondary text #d2bdc3, action #e8c37e and
warning #ff453a. System sans-serif fonts avoid external font requests. A large
Start opens three numbered voice choices, followed by camera or destination
controls. Left-aligned instructions, visible focus, zoom and scroll, persistent
Stop and Repeat, and a concise spoken-message transcript take precedence over
decoration. The numbers correspond to spoken mode commands.

## Milestones 2–5: backend, mobile camera, HTTPS, voice and state machine
Implemented: retained FastAPI and camera capture; added three-mode workflow,
button/text alternatives, persistent transcript/Stop/Repeat, navigation without
camera permission, configurable voice/frame timings, and an HTTPS launcher.
Files changed: server/main.py, server/config.py, web/index.html, web/workflow.js,
.env.example, scripts/run_https.ps1.
Tests run: browser Start/menu/search/navigation checks; JavaScript command/state
tests; syntax check; mobile-width screenshot review at 390 by 844 pixels.
Tests passed: Start opens all three modes, query and destination prompts appear,
Stop reports camera/microphone off, browser error log is empty.
Issues found: asynchronous camera and analysis responses could outlive mode changes;
the prior interface had no three-mode selection or continuous keyword workflow.
Issues fixed: generation guards, request cancellation, fresh session IDs, stopped
tracks, loop cancellation and continuous search. The viewport override was reset.
Next milestone: inference, geometry and priority integration.

## Milestones 6–10: oriented text, OCR, objects and spatial estimates
Implemented: retained CRAFT/EasyOCR and optional YOLO-OBB rectification; preserved
quad coordinates and combined detection/OCR confidence. Added oriented line joining
for adjacent OCR words. Retained replaceable object models and schema validation.
Files changed: server/text_pipeline.py, server/detector.py, server/interfaces.py,
server/spatial.py, tests/test_prototype.py, scripts/smoke_test.py.
Tests run: pure geometry tests, oriented-region regression, real horizontal and
12-degree tilted sign inference; real object inference on an existing dataset frame.
Tests passed: both generated signs recognized as EXIT ROOM 205 and matched room 205;
quads retained, sign steps null. Object inference returned a person detection.
Issues found: tilted OCR returned separate words in reverse reading order; signs
could receive floor-plane distance estimates; distances rounded up.
Issues fixed: oriented baseline grouping (separate rows stay separate), no metric
steps for text/signs/stairs, finite/valid geometry guards and conservative rounding.
Next milestone: summary/search/navigation and safety integration.

## Milestones 11–15: priority, environment, search, navigation and integration
Implemented: bounded priority speech queue, duplicate cooldown, two-observation
confirmation, cautious multi-clue context summaries, nearest relevant text ranking,
whole-token matching with restricted fuzzy matching and exact numeric tokens.
Navigation has validated coordinates and an explicit Maps fallback, no invented
distance/time. Inference errors preserve available components and disclose failures.
Files changed: server/priority.py, server/environment.py, server/symbols.py,
server/navigation.py, server/main.py, server/upload_limit.py, web/workflow.js.
Tests run: priority, search, API validation, invalid images, chunked oversized bodies,
degraded inference, expired tracking, navigation and workflow regressions.
Tests passed: tests cover men/menu/women, room 205/206, split OCR words, cooldown,
critical queue precedence, missing coordinates and component failure.
Issues found: old cached text coordinates could be spoken after camera movement;
component exceptions aborted all inference; multipart streams needed a length-free cap.
Issues fixed: skipped OCR emits no stale regions; independent component handling;
bounded multipart streams and decoded image dimensions; confirmed search sources.
Next milestone: final tests, performance observations and documentation.

## Milestones 16–18: verification, performance and documentation
Implemented: extended Python tests, Node state/queue tests and CI; rotated-IoU/F1 and
matched OCR accuracy/CER/WER evaluation; real-model smoke script; rewritten README
with setup, model, dataset, HTTPS, API, privacy and research limitations.
Files changed: tests/, scripts/evaluate_text.py, scripts/smoke_test.py,
.github/workflows/tests.yml, requirements.txt, README.md and this log.
Tests run: full pytest suite, Node test runner, inline JavaScript syntax, git diff
whitespace check, real inference smoke, local server health and browser workflow.
Tests passed: **174 Python tests**, **2 JavaScript tests**. One existing FastAPI
TestClient dependency deprecation warning remains. No script errors in browser check.
Issues found: Windows sandbox blocked pytest temporary directories. A browser action
was later blocked by automatic approval review because the account usage limit was
reached. The user resumed on 11 September.
Issues fixed: pytest ran successfully with authorized temporary-directory access;
the resumed browser Stop/error-log/viewport-reset checks completed successfully.
Next milestone: user-operated phone and physical safety evaluation; custom model
training/evaluation where the required reviewed labels and hardware are available.

## Actual measurements and their limits
The final real-model smoke run returned horizontal and tilted sign results in
2,123 ms and 2,159 ms respectively (runs/prototype-smoke.json). An earlier run under
different load took 15,945 ms for the horizontal image. These are individual
integration observations, not a benchmark or guaranteed latency. Real object
inference on one existing frame took about 8.4 seconds including lazy loading.
No mAP, accuracy, camera-to-speech latency or mobility-safety performance is inferred
from these fixtures. Training was not run during this implementation.

## Remaining external validation
- Actual phone camera/microphone/GPS permission behavior and HTTPS certificate trust
  require testing on the user's device. No certificate trust was installed here.
- No routing credentials/provider are configured; Google Maps resolves the handoff.
- No custom text OBB weights are present; CRAFT is the active oriented fallback.
- Candidate AV-6 weights remain undeployed; COCO cannot detect all requested hazards.
- Distances depend on assumed camera/object geometry and require physical validation.
- Live voice recognition depends on browser support and may use an external service.
- The server is for trusted local use; public production deployment needs authentication.

## September 12, 2026 — real-time and voice lifecycle repair

Extracted browser orchestration into app.js, speech.js and realtime.js. Unified
recognition and command routing, removed broad click-to-speak handlers, repaired
post-TTS listening and transient recognition restart handling. Added persistent
tracking, independent latest-frame object/OCR lanes, bounded location handoff,
current-state indicators, and camera/error recovery. See REALTIME_FIXES.md for root
causes, affected files, test coverage, measurements and remaining device limits.

Validation: 22 JavaScript regressions pass; backend suite and actual OCR smoke
results are recorded in REALTIME_FIXES.md. Browser checks found and fixed native
timer binding; physical camera/microphone success is not claimed.
