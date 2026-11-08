# Privacy architecture

§16 requires that every perception operation state whether the camera frame
stays on the device or leaves it. This is that statement, for the build as it
exists, not for the one intended.

## Where frames go today

| operation | frame stays local? | detail |
|---|---|---|
| object detection, both providers | **NO — leaves the device** | canvas → JPEG q0.85 → `POST /api/detect/objects` |
| text detection and OCR | **NO — leaves the device** | same capture path, `POST /api/detect/text` |
| tracking, world model, announcement policy | **yes** | runs in `realtime.js` in the browser |
| spatial estimates, risk ordering | NO | computed server-side today from the uploaded frame |
| vertical-FOV calibration | yes | `localStorage` only |
| speech synthesis | yes, text only | no audio leaves the device |
| speech recognition | **NO — audio leaves the device** | `webkitSpeechRecognition` sends audio to the browser vendor's service. This is the platform's behaviour and cannot be configured away; it is why §25's fallback path matters. |
| geolocation | NO, when a route is requested | coordinates sent for navigation only |

**So: in the current build, essentially all visual perception uploads the
camera frame.** The architecture intends to move the safety loop into the
browser, which would change most of this table — but it has not moved, and the
table is written for what runs.

## Rules for the server path, while it exists

- HTTPS only. No frame over plain HTTP, ever.
- Authenticated requests. Not yet implemented — the API is currently open, which
  is acceptable for a prototype behind a temporary tunnel and is **not**
  acceptable for anything deployed (§38).
- **No frame persisted by default.** Frames are processed in memory and
  discarded. Any debugging that writes frames to disk is opt-in, local, and off
  in the default configuration.
- No frame in a log line, no frame in an error report, no frame in telemetry.
- Upload size limits and rate limits enforced server-side
  (`server/upload_limit.py` exists; rate limiting does not).

## Personal memory

Browser-local, in IndexedDB, when it is built. Never synchronised to a server
without the user asking for it in so many words.

It may hold user labels, remembered places, familiar objects, corrections and
preferences. Remembered places are location history, which is among the most
sensitive data a blind user could hand to an application, so:

- the user can list, edit and delete every entry, and wipe everything;
- nothing is recorded as a place unless the user said to remember it;
- no continuous location trail is accumulated as a side effect of navigation.

Note the platform constraint from
[BROWSER_MATRIX.md](../05-testing/BROWSER_MATRIX.md): on iOS, IndexedDB for a
site that was not added to the home screen is evicted after about seven days.
Personal memory is therefore durable only for installed PWAs, and the
onboarding has to say so rather than silently losing a user's places.

## What the user must be told, in the app

Before the first camera frame: that frames are sent to a server for processing
while the current build is in use, that they are not stored, and how to stop.
Permission onboarding is the place for this, and it has to be reachable by
screen reader without sighted help.
