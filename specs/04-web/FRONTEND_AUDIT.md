# Audit of the existing web and server code

Done before any rewrite, as §46 requires. Every component is classified
KEEP / REFACTOR / REPLACE / DELETE with the reason, and nothing is marked for
replacement because a newer framework exists.

## What is actually there

No build step, no `package.json`, no TypeScript, no bundler. `web/` is five
hand-written ES modules — 1,130 lines — loaded directly by `index.html`, tested
by four Node suites that import them without a build. The server is FastAPI,
~2,000 lines, with 230 passing Python tests.

That absence of tooling is worth stating plainly, because it cuts both ways: the
project has no build to maintain and no dependency tree to audit, and it also
has no type checking, no PWA, and no way to run anything off the main thread.

## Client

| component | lines | verdict | why |
|---|---|---|---|
| `web/realtime.js` — `DetectionMemory` tracking, announcement gating | 111 | **KEEP** | This is the behavioural fix from the last iteration and its acceptance criteria are pinned by tests. It is framework-free and portable as it stands. Moving it to a Web Worker later needs no change to its logic. |
| `web/speech.js` — sentence construction | 179 | **KEEP** | Pure string building over a detection list. No browser coupling. |
| `web/workflow.js` — scan/search state machine | 52 | **KEEP** | Small, tested, no coupling. |
| `web/app.js` — camera, capture, fetch, speech output, UI, orientation | 540 | **REFACTOR** | Not wrong, but it is the one file that does everything, and it is where every web requirement in this brief lands: worker offload, keep-latest frame semantics, PWA lifecycle recovery, an in-browser inference path. Split it along the boundaries it already has internally — capture, transport, output, UI — rather than rewriting it. |
| `web/index.html` — accessible shell, live regions | 248 | **REFACTOR** | Accessibility semantics are already present and must not be lost. Needs a manifest link, install onboarding, and a permission-recovery path. |
| `web/check.cjs` | — | **DELETE** | A browser script run under Node by mistake; it fails with `ReferenceError: location is not defined`. It is already gitignored. The real tests are `tests/*.cjs`. |

**No hardcoded server address anywhere.** Every call is a relative path
(`/api/detect/...`, `/api/config`, `/api/health`), so the client already works
against whatever origin serves it. §37's laptop dependency is therefore a
*deployment* gap, not a code gap — the fix is somewhere to deploy, not a change
here.

## Server

| component | verdict | why |
|---|---|---|
| `server/perception.py` — providers, canonical vocabulary, authority, fusion | **KEEP** | The current architectural baseline (R-003). Its interface is also the right interface for a browser runtime: frame in, normalised detections out. |
| `server/spatial.py`, `priority.py`, `symbols.py` | **KEEP** | Geometry and ordering. No server dependency; these are candidates to move into the browser unchanged. |
| `server/classes_av.py` | **KEEP** | Vocabulary and the `stairs_down` safety wording. |
| `server/text_pipeline.py` — OCR | **KEEP** | EasyOCR is Python-only, so this stays server-side until a browser OCR path is measured (§21). |
| `server/detector.py` | **REFACTOR** | Now a thin shim over `perception`. Fold it in once nothing else imports it. |
| `server/main.py` — FastAPI app | **REFACTOR** | Correct, but unauthenticated and unrate-limited, which §38 does not allow for anything deployed. |
| `server/demo.py`, `server/uber.py` | **KEEP, out of scope** | Demo scaffolding. Untouched by this direction change. |

## What does not exist yet

Nothing to audit, everything to build: web app manifest, service worker, offline
shell, model-asset caching, Web Workers, IndexedDB personal memory, in-browser
inference, TypeScript, any deployed environment.

## Consequence for the plan

The useful finding is that the logic worth keeping — tracking, announcement
policy, sentence construction, spatial reasoning, fusion — is all
framework-free and all portable to a browser runtime. What has to be built is
the *platform* around it: a build, a worker boundary, a PWA shell, and somewhere
to deploy. A rewrite would throw away the tested part and keep the part that
does not exist yet.
