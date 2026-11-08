# Product requirements

Web application and installable PWA. State column is honest: most of this is not
built, and the few things that are have the test or measurement that says so.

## Platforms

| requirement | state |
|---|---|
| Chrome on Android — primary experience | partial — runs, never tested end to end on a phone |
| Safari on iOS — where browser capability allows | **not tested**; see BROWSER_MATRIX for what iOS cannot do |
| tablets | not tested |
| desktop browsers — development and testing only | works |

No requirement is met by "it loads". A capability counts when it has been
exercised on that platform.

## Functional

### Perception
- Detect general objects (vehicles, people, street furniture) and
  assistive-specific objects (doors, stairs, bins, chairs) **at the same time**,
  from several providers fused into one stream — **done** (R-003).
- Read text in the scene and associate it spatially — **partial**, server-side.
- Estimate distance and direction for each entity — **partial**: direction is
  sound, distance is an estimate from box geometry, not a measurement.
- Detect motion and approach — **done**, drives the approach announcements.
- Identify walkable surface, kerbs and ramps — **not built**, and needs
  segmentation rather than detection.
- Describe a scene on request — **not built**.

### Voice
- Operate by voice with minimal visual interaction: "What's around me?",
  "Where is the door?", "Read this.", "Find EXIT.", "Repeat.", "Stop.",
  "Guide me.", "Remember this.", "Forget this.", "Describe this room."
  — **partial**: scan, search and repeat exist; the rest do not.
- A clear non-voice path wherever recognition is unavailable or fails —
  **required**, not built. Browser speech recognition is unreliable on every
  platform, so this is a primary path, not a fallback.

### Speaking
- One speech authority; no module speaks independently — **done**.
- Speak only when information is new, risk changed, the user asked, guidance
  state changed, or a critical event occurred — **done**, pinned by
  `tests/test_realtime.cjs`.
- Never return to frame-by-frame narration — **binding**.
- Critical events interrupt — **done**.
- Degrade audibly when speech fails — **not built**.

### Safety
- Risk overrides route guidance — **partial**.
- Never say a road is safe to cross — **binding**.
- Never claim a staircase's direction the detector cannot determine —
  **done** (R-005).
- Announce camera or perception loss rather than falling silent — **not built**.

### Navigation
- Macro (destination, route, maps) separated from micro (obstacles, doors,
  stairs, people, surface), combined by a context engine — **partial**:
  `server/navigation.py` exists, the context engine does not.
- Document which features need the internet — **required**.

### Memory
- Remember places, labels, corrections and preferences, device-local.
- The user can view, edit, delete and reset all of it.
- No silent accumulation of location history.
— **none of this is built.**

## Non-functional

| requirement | target | state |
|---|---|---|
| installable PWA with offline shell | — | **not built** |
| no expensive work on the main thread | no frozen controls, no interrupted speech | **not met** — no workers exist |
| frame handling | keep-latest, queue of one | **not implemented** |
| end-to-end latency | p50 and p95 recorded per stage | **never measured** |
| thermal and battery behaviour on a phone | recorded | **never measured** |
| no production dependence on a developer machine | — | **not met** — Cloudflare quick tunnel to a laptop |
| HTTPS, authenticated API, rate limits, CSP | — | **not met** — API is open |
| screen-reader operable without sight | TalkBack and VoiceOver | **not tested** |
| camera frames: stated local or remote per operation | — | **done** — PRIVACY_ARCHITECTURE.md |

## Quality bar for perception, which gates the rest

Metrics against a **locked test set**, with per-class recall and explicit false
negatives for safety classes. Not summarised by mAP alone.

Current: fused recall **0.297** on validation data, no locked test set, outdoor
capability unmeasured. This is below any reasonable bar for an application a
blind user would walk behind, and improving it takes precedence over further UI
work (§40).
