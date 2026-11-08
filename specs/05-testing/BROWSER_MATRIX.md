# Browser capability matrix

The product is a browser/PWA application, so every capability it depends on is a
browser capability with a different answer on each platform. This file records
those answers. It is written from documented platform behaviour; the **Verified**
column says whether *this project* has observed it on a real device, and today
most of it says no.

Legend: **YES** works · **PARTIAL** works with a caveat that changes the design ·
**NO** not available · **PERM** needs a permission prompt · **GESTURE** needs a
user gesture first · **HTTPS** secure context only.

## Primary target: Chrome on Android. Secondary: Safari on iOS.

| capability | Chrome Android | Edge Android | Safari iOS | Chrome desktop | Verified here |
|---|---|---|---|---|---|
| `getUserMedia` camera | YES · PERM · HTTPS | YES · PERM · HTTPS | YES · PERM · HTTPS · GESTURE | YES · PERM · HTTPS | desktop + Android |
| rear camera (`facingMode: environment`) | YES | YES | PARTIAL — honoured, but `width/height` ideals are frequently ignored | PARTIAL — often no rear camera | desktop only |
| microphone | YES · PERM | YES · PERM | YES · PERM · GESTURE | YES · PERM | no |
| `SpeechRecognition` | PARTIAL — `webkitSpeechRecognition` only; sends audio to Google, so it needs the network and is not private or offline | PARTIAL — as Chrome | PARTIAL — `webkitSpeechRecognition` since 14.5; needs a gesture per session and ends often | PARTIAL — as Chrome | no |
| `speechSynthesis` | PARTIAL — `getVoices()` is empty until `voiceschanged`; utterances are dropped while the tab is hidden | PARTIAL — as Chrome | PARTIAL — first utterance needs a gesture; queue is cleared when the page is hidden | YES | desktop only |
| TTS while recognition is live | PARTIAL — TTS audio re-enters the microphone unless recognition is paused | PARTIAL | PARTIAL | PARTIAL | no |
| `navigator.vibrate` | YES | YES | **NO** — never shipped on iOS | NO (no hardware) | no |
| Geolocation | YES · PERM · HTTPS | YES · PERM · HTTPS | YES · PERM · HTTPS | YES · PERM · HTTPS | no |
| `deviceorientation` | YES · HTTPS | YES · HTTPS | PARTIAL — needs `DeviceOrientationEvent.requestPermission()` from a gesture | PARTIAL — usually no sensors | no |
| WebAssembly | YES | YES | YES | YES | no |
| WASM SIMD | YES | YES | YES (16.4+) | YES | no |
| WASM threads | PARTIAL — needs COOP/COEP response headers, which cross-origin model CDNs break | PARTIAL | PARTIAL | PARTIAL | no |
| WebGPU | YES (121+) | YES | PARTIAL — Safari 26 / iOS 26; flagged before that | YES | no |
| IndexedDB | YES | YES | PARTIAL — 7-day eviction for sites not installed to the home screen | YES | no |
| Service worker | YES | YES | YES, but its cache is subject to the same eviction | YES | no |
| PWA install | YES — WebAPK, `beforeinstallprompt` | YES | PARTIAL — Add to Home Screen only; no install prompt API, so onboarding must be instructions | YES | no |
| Wake Lock (`screen`) | YES | YES | YES (16.4+) | YES | no |
| **camera while backgrounded or screen-locked** | **NO** | **NO** | **NO** | NO | — |

## The limitation that shapes the product

**A browser cannot keep the camera running when it is not in the foreground.**
When the screen locks, the user switches apps, or the OS reclaims memory, the
video track ends or the page is frozen, and on iOS the page may be discarded
entirely. There is no web API that changes this, and no amount of service-worker
work gets around it: a service worker has no camera.

This is not a bug to fix. It is a property of the platform the product owner has
chosen, and the consequences must be designed for rather than worked around:

- **Guidance is a foreground activity.** The user holds the phone, with the app
  on screen, for as long as they want perception. Say so in onboarding instead
  of letting them discover it at a kerb.
- **Hold the screen awake while scanning** (Wake Lock) so the common case — the
  phone in a hand, untouched for a minute — does not end the session.
- **Returning to the foreground must recover, not resume.** The track is dead
  and the world model is stale. On `visibilitychange` to visible: re-acquire the
  camera, clear tracked entities rather than re-announcing them, and say one
  short line that scanning has restarted. A world model carried across a gap of
  unknown length describes a room the user may have left.
- **Announce the loss.** If the track ends while the user believes they are being
  guided, silence is the dangerous failure. Speak it.

## What must be measured on a real device before any of this is settled

Everything in the Verified column that says no. Specifically: whether Chrome
Android speech synthesis survives a long session, whether recognition restarts
reliably after TTS, what the camera actually delivers for resolution and frame
rate on a mid-range phone, and what sustained inference does to device
temperature. §25 and §31 of the current brief ask for these; none has been run.
