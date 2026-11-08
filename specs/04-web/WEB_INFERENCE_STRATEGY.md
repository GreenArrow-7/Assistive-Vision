# Where inference runs

§14, §15, §30 and §42 of the brief ask for in-browser inference, server
inference and a hybrid to be researched and chosen on evidence. This file holds
the evidence that exists so far, and is explicit that it is not yet enough to
choose.

## What runs today: Strategy B, entirely

The browser draws the video frame to a canvas, encodes it as JPEG at quality
0.85, and POSTs it to `/api/detect/<lane>`. Both detectors run in Python on the
server. **Camera frames leave the device on every processed frame.** That is the
current, unavoidable privacy position, recorded in
[PRIVACY_ARCHITECTURE.md](../02-architecture/PRIVACY_ARCHITECTURE.md).

Measured server-side cost: 42 ms for one provider, **87 ms for both fused**
([MODEL_EVALUATION.md](../../docs/MODEL_EVALUATION.md)). Network time is on top
of that and has never been measured on a mobile connection.

## Measured: the models export cleanly, and ONNX Runtime is faster than PyTorch

Exported with `scripts/` tooling at imgsz 640, opset 12, simplified, static
shapes. Latency is ONNX Runtime 1.31.1 on the CPU execution provider, desktop
Intel i5-10500, 15 runs after 3 warm-ups.

| model | PyTorch CPU p50 | ONNX Runtime CPU p50 | p95 | file | gzipped |
|---|---|---|---|---|---|
| AV-7 (7 classes) | 42.2 ms | **27.6 ms** | 32.7 ms | 12.27 MB | 10.54 MB |
| COCO yolov8n (80 classes) | 45.1 ms | **34.6 ms** | 38.5 ms | 12.85 MB | 11.06 MB |

Two findings worth carrying forward:

**Export is not a penalty, it is a speedup.** ONNX Runtime is 35% and 23%
faster than PyTorch on the same CPU. The usual worry about leaving `.pt` behind —
that the exported graph will be slower — is measurably false here, so §30's
requirement not to treat `.pt` as the only deployable format costs nothing.

**The download is the real cost of in-browser inference.** Both providers is
**25.1 MB raw, 21.6 MB gzipped**, before the app can see anything at all. On a
mid-range Android phone on mobile data that is a first-run wait measured in
tens of seconds, and under iOS Safari's 7-day eviction for non-installed sites
it may be paid repeatedly. Cache Storage and an installed PWA mitigate it;
nothing removes it.

**What these numbers are not.** They are desktop x86 with native ONNX Runtime.
A browser runs WASM or WebGPU, not native ORT, and a phone's CPU is several
times slower than an i5-10500. Treat 27.6 ms as a *floor* that no browser
runtime will beat, not as an estimate. The only honest number is one measured in
Chrome on an actual phone, and nobody has one.

## The three strategies, and what each still needs

**A — in-browser.** Frames never leave the device, which answers §16 completely,
and the app keeps working offline. Needs: a WASM/WebGPU runtime chosen by
benchmark (not popularity — §3), the 21.6 MB download solved, and sustained
thermal behaviour measured. Unknown and important: whether a phone can hold a
useful frame rate for minutes without throttling.

**B — server.** What exists. Stronger models, instant updates, no download, and
a phone that stays cool. Costs: every frame uploaded, dependence on
connectivity at exactly the moment a user is walking somewhere unfamiliar, and
a server bill proportional to use. Also currently violates §37: the only
deployment is a developer machine behind a Cloudflare quick tunnel.

**C — hybrid.** The one that matches the product's actual shape, and the reason
is not architectural neatness, it is the failure mode. Hazard detection must not
stop working because a tunnel dropped the signal, so the safety loop belongs on
the device. Scene description and conversational reasoning are requested
explicitly, tolerate a second of latency, and are the things a server is
genuinely better at.

```
browser : camera -> small detector -> tracking -> world model -> risk -> speech
                                         |
                          on request     v
server  :                     scene description, OCR, reasoning
```

Note what this implies for the thing already built: `perception.py`'s interface
— frame in, normalised detections out — is the same interface a browser runtime
needs, and `realtime.js`, `speech.js` and `spatial.py` are already
framework-free. The hybrid split follows the seam the code already has.

## Not decided, and why

No strategy is chosen here. The deciding measurement — inference latency, memory
and thermal behaviour in Chrome on a real Android phone — has not been taken,
and §42 forbids promoting a model on desktop Python numbers alone. Choosing now
would be choosing on the basis of the two platforms the product does not ship
on.

What makes it decidable, in order: export the models (**done**), get a deployed
origin so the browser is not talking to a laptop, run the same 336-image
evaluation through a browser runtime to confirm the exported graph scores
identically, then measure on a phone.
