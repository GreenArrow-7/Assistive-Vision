# Voice-first and real-time fixes — September 12, 2026

## Root causes and changes

| Problem | Root cause | Implemented change |
|---|---|---|
| Start did not scan | Welcome Start routed to the menu; parser aliases differed across routers | One parser maps Start/Start scan/Start scanning/Begin scanning to monitoring |
| Self-recognized TTS | Separate gate and command recognizers, uncancelled restart timers, no result barrier during speech | One recognizer, stop/end barrier, actual utterance end/error events, cooldown, delayed-result guard |
| Repeated objects/text | Speech-string cache expired after nine seconds while detections stayed present | Persistent track identity, last-spoken geometry, stable count and reappearance lifecycle |
| Old boxes/speech | Error/blur paths skipped render cleanup; cache expiry did not invalidate pending speech | Current-frame replacement, misses/expiry, per-lane invalidation, queue validity checks |
| Lag | One inference lock covered both object model and expensive OCR; combined response waited for both | Independent model locks and bounded latest-frame browser lanes; immediate load shedding |
| Manual camera instructions | Blur threshold returned before any inference | Quality metadata with continued inference and conditional OCR contrast adjustment |
| OCR fragility | Already rectified OBB crops ran another text detector; uncertain orientation and garbage were unfiltered | Recognition-only OBB crops, bounded uncertain-region rotation retries, garbage filtering; existing quad/line joining retained |
| Search confusion | Unknown phrases fell through to search; bare phrases accepted indefinitely; broad emergency-exit synonyms | Explicit intents, bare target only after a prompt, whole-token matching, stricter emergency-exit synonyms |
| Navigation stalled | GPS relied only on browser timeout and handoff only displayed a link | Independent GPS deadline and errors, same-tab automatic handoff after TTS completion |
| Distance and time failed | Destination not retained as separate state; phrase parsed as search | Persistent destination and NAV_INFO intent |
| Repeated environment summary | Scan was mapped to continuous full summaries | One summary over current lane results, then change-only monitoring |
| Delayed hazard warning | Every live hazard needed two frames | Confidence-gated immediate path for approximate very-close hazards and supported stairs |
| Misleading status | Several booleans and old responses populated indicators | Explicit operation state, real media track status, actual speech lifecycle and per-lane processing status |
| Browser-only startup exception | Native browser timer functions received a class instance as receiver | Detached timer wrappers plus regression test |

No new runtime library or model is introduced. FastAPI, the current detector schema,
CRAFT/EasyOCR, optional YOLO-OBB path, spatial calculations, accessible HTML, dataset
tools and existing evaluation remain. JavaScript orchestration is extracted from the
large inline script into one entry point, removing the ignored duplicate inline block
inside the old external-script tag and the separate speech routers.

## Changed files
- `web/index.html`: existing UI preserved; direct Start, persistent voice control and script entry points.
- `web/app.js`: camera lifecycle, independent inference lanes, current overlays, search, one-time scan, navigation and status integration.
- `web/workflow.js`: centralized command parsing and retained destination.
- `web/speech.js`: one recognition instance and bounded, event-synchronized TTS manager.
- `web/realtime.js`: detection memory, latest-frame scheduling, GPS deadline and Maps URL builder.
- `server/main.py`: split inference, model locks, timings, quality metadata, critical path and static module serving.
- `server/config.py`, `.env.example`: adjustable timings, expiry and CPU thread budget.
- `server/text_pipeline.py`: conditional contrast, recognition-only crops and text filtering.
- `server/priority.py`, `server/navigation.py`: immediate warnings, no steering prompts, Maps directions response.
- `tests/test_app.cjs`, `test_speech.cjs`, `test_realtime.cjs`, `test_workflow.cjs`: deterministic frontend regressions.
- `tests/test_live_pipeline.py`: actual API/concurrency/quality regressions with test doubles.
- `.github/workflows/tests.yml`, `README.md`, `docs/IMPLEMENTATION_LOG.md`: CI and operating/validation documentation.

## Verification
The original 174 Python tests passed after the initial changes. With new backend
regressions, 178 Python tests pass. The expanded JavaScript suite covers the actual
app entry point using explicit DOM/media/API test doubles, plus speech and tracking
lifecycle behavior. These tests do not simulate a successful physical camera trial.

Covered workflows: welcome -> Start -> both inference lanes; TTS stop/end/cooldown
and delayed callbacks; ten unchanged chair frames; disappearance and reappearance;
unchanged EXIT; safe keyword parsing; navigation and retained destination; cancelled
handoff; one-time environment summary; critical precedence; single-flight restart;
stale result rejection; GPS timeout/denial; camera permission failure; blurred-frame
inference; OCR garbage filtering; object completion while OCR is blocked.

Real-model smoke verification recognized **EXIT ROOM 205** on both horizontal and
12-degree tilted generated signs. The observed combined request times were 3105 ms
and 2677 ms in this run. These are individual local CPU observations, not an FPS,
field accuracy, or camera-to-speech benchmark. The test asserts real model results;
no detection or distance is hardcoded as a real result.

The browser test exposed and fixed a native timer receiver error that Node had not
exposed. Keyboard Start reached camera startup. This browser reported unavailable
speech output and did not supply a usable camera stream in that check. Browser
policy/device limits remain distinct from deterministic test success.

## Remaining limitations
- Autoplay and microphone permissions can require a first tap. The app attempts
  automatic welcome/listening and retains accessible fallback controls.
- Physical speaker-to-microphone echo performance still requires device testing;
  lifecycle/late-event prevention is covered by deterministic regressions.
- Severe blur, low light, occlusion and missing classes can cause missed detections.
- Tracking is label/geometry based. It cannot guarantee identity across abrupt motion.
- Monocular distances remain approximate; text/sign/stair distances are withheld.
- COCO remains the default. No custom OBB weights were added and AV-6 candidates were
  not promoted; full hazard coverage and model training remain outstanding.
- Maps resolves destination and route metrics. The app cannot verify arrival in an
  external Maps app, internet availability there, or destination ambiguity.
- Slow machines may exceed the response-age limit; increase it only after measuring
  the tradeoff between useful recognition and stale information.

## Run
```powershell
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 127.0.0.1 --port 8001
.\.venv\Scripts\python.exe -m pytest tests -q
node --test --test-isolation=none tests/test_*.cjs
.\.venv\Scripts\python.exe scripts/smoke_test.py
```
Open `http://localhost:8001`. Port 8000 was occupied during this verification, so the
updated v1.9 verification server uses 8001. For a phone, follow README's trusted LAN
HTTPS setup and load `.env` with `--env-file .env` if configuration overrides are used.

## High-priority voice follow-up
Passive welcome-screen and camera-preview click handlers were removed; those areas
no longer act as speech controls. The camera preview is an accessible region.
Recognition resumes from the owned utterance completion event plus cooldown, rather
than the browser's sometimes-lagging `speechSynthesis.speaking` flag. Transient
`InvalidStateError` startup conflicts retry on the same recognizer. A delayed stop
completion retains listening intent without creating a concurrent recognizer.
Visibility changes no longer stop the voice session; hidden-page inference is
paused. Explicit Stop and page exit still release the session.

All 22 JavaScript regression tests pass, including four consecutive commands after
one Start click, delayed native speech state, transient recognition startup races,
passive click handlers, visibility changes, and intentional Stop. Real microphone
recognition and acoustic echo still need testing in the target browser and device.
