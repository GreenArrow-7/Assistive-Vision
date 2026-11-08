# System architecture

Target shape for the web/PWA product. Marked against what exists today, so the
diagram cannot be mistaken for a description of the current build.

```
                 Mobile browser / PWA
                           |
   camera · microphone · geolocation · orientation · user commands
                           |
                           v
                  frame / sensor pipeline          [exists: app.js]
                           |
                           v
                   perception manager              [exists: perception.py, server-side]
                  /        |         \
          objects       OCR/text    depth/spatial
          [exists]     [exists]      [NOT BUILT]
                  \        |         /
                           v
                        tracker                     [exists: realtime.js]
                           v
                 spatial world model                [partial: realtime.js tracks; no entity schema]
                           v
                      risk engine                   [exists: priority.py, server-side]
                           v
                     context engine                 [NOT BUILT]
                      /         \
              navigation       memory
            [partial: navigation.py]  [NOT BUILT]
                      \         /
                           v
                 grounded AI brain                  [NOT BUILT]
                           v
                announcement policy                 [exists: realtime.js gating + speech.js]
                           v
            speech · haptics · accessible UI        [exists: app.js, index.html]
```

## Where each stage runs

Today: everything from perception to risk runs on the server; tracking,
announcement policy and output run in the browser. The intended end state is the
hybrid in [WEB_INFERENCE_STRATEGY.md](../04-web/WEB_INFERENCE_STRATEGY.md) —
the safety loop on the device, optional reasoning on a server — and it is not
yet chosen, because the deciding measurement has not been taken.

## Technology direction

TypeScript, browser APIs, a service worker and a web app manifest, Web Workers
for anything expensive, IndexedDB for personal memory, Cache Storage for the app
shell and model assets, WebAssembly or WebGPU for browser inference. None of
this exists yet: there is no build step, no `package.json` and no TypeScript in
the repository today — see [FRONTEND_AUDIT.md](../04-web/FRONTEND_AUDIT.md).

An inference runtime is chosen by benchmark, not by popularity (§3).

## Rules that hold across the whole system

**One speech authority.** Candidates flow through a single announcement policy
into a single queue. No module speaks on its own. This is what stopped the
repeat-announcement problem and must not be reintroduced by a new module that
finds it convenient to call `speechSynthesis` directly.

**Nothing expensive on the main thread.** Inference, decoding and fusion go to a
worker. Frozen controls, a delayed live region or interrupted speech are
accessibility failures, not performance ones.

**Keep-latest frame semantics.** A bounded queue of one. A backlog of stale
frames describes where the user was, and for this product a late answer is a
wrong answer.

**Model names live in configuration.** `config.PERCEPTION_PROVIDERS` and nowhere
else. The world model and risk engine never name a model (R-003).

**Recovery, not resume, on return to foreground.** The browser cannot keep the
camera while backgrounded. Re-acquire the camera, discard the world model rather
than re-announcing it, and say that scanning restarted. See
[BROWSER_MATRIX.md](../05-testing/BROWSER_MATRIX.md).

**No laptop in the production path.** The browser talks to deployed
infrastructure or to itself. A developer machine on the same Wi-Fi is a
development arrangement, and the current Cloudflare-tunnel setup is exactly that
(§37, open).

## Server role, bounded

A server is allowed, for: an authenticated API, optional inference, advanced
reasoning, model metadata and distribution, and navigation integrations. It is
not allowed to be required for the immediate safety loop, and it is not allowed
to be a developer's laptop.
