# Definition of done

Web requirements. The previous Android-specific criteria are removed (R-007) —
there is no APK, no Gradle build and no Play Store listing in this definition.

Nothing here is met today. The column is honest on purpose.

## Web build

| criterion | state |
|---|---|
| clean install from a fresh checkout succeeds | **no** — there is no `package.json` |
| production build succeeds | **no** — there is no build step |
| lint and type checks pass | **no** — no TypeScript, no linter |
| automated tests pass | **yes** — 230 Python, 4 Node suites |

## PWA

| criterion | state |
|---|---|
| web app manifest valid | **no** — does not exist |
| service worker registers and serves the offline shell | **no** |
| installable on Chrome Android; documented instructions for iOS | **no** |
| model assets cached, with a versioned update path | **no** |
| application version visible and reportable | **no** |

## Camera

| criterion | state |
|---|---|
| works on a real mobile browser | **no** — desktop and Android browser only, never on a phone end to end |
| permission denial produces a clear, screen-reader-reachable recovery path | partial |
| orientation change handled without losing the session | **unverified** |
| return from background re-acquires the camera and discards stale state | **no** — see BROWSER_MATRIX |
| camera loss is announced, not silent | **no** |

## Perception

| criterion | state |
|---|---|
| required providers load reliably, and zero providers fails loudly | **yes** — `perception.warmup()` |
| metrics documented against a **locked test set** | **no** — validation only; `assistive_test_v1` not built |
| per-class recall and false negatives reported for safety classes | **yes** — MODEL_EVALUATION.md |
| vehicle and outdoor capability measured on project data | **no** — asserted from vocabulary only |
| browser inference performance documented on a target phone | **no** |

## Speech

| criterion | state |
|---|---|
| duplicate announcements controlled | **yes** — pinned by `tests/test_realtime.cjs` |
| critical events interrupt | **yes** |
| "repeat" works | **yes** |
| degraded behaviour when speech synthesis or recognition fails | **no** |
| verified through a long session on Chrome Android | **no** |

## Accessibility

| criterion | state |
|---|---|
| ARIA semantics and live regions correct | partial — present in `index.html`, never audited |
| focus order correct | **unverified** |
| fully operable without sight | **unverified** — not tested with TalkBack or VoiceOver |
| permission errors and recovery announced | partial |

## Privacy

| criterion | state |
|---|---|
| camera-frame handling documented | **yes** — PRIVACY_ARCHITECTURE.md |
| user told, before the first frame, that frames are uploaded | **no** |
| personal memory viewable, editable, deletable | **no** — does not exist |

## Security

| criterion | state |
|---|---|
| web security review complete | **no** |
| API authenticated and rate-limited | **no** — currently open |
| no secrets in client JavaScript | **yes** |
| model asset integrity verified on download | **no** |

## Testing

| layer | state |
|---|---|
| unit | **yes** |
| integration | **yes** |
| browser (automated, real browser) | **no** |
| mobile browser | **no** |
| perception metrics against a locked test set | **no** |
| safety scenarios | **no** — the real-world test matrix is unwritten |
| accessibility | **no** |

## The one that gates all the others

Fused recall is **0.297**. A product that misses two thirds of the obstacles
annotated in front of it is not done regardless of how many boxes above turn
green, and a quiet application with poor recall is not an improved one (§16,
§40). Perception quality is the first criterion, not the last.
