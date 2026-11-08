# Project state

Updated 2026-11-08. Claims here are measured or cite the file that backs them.

**Product direction changed this iteration.** The end-user application is a
**web application and installable PWA** and nothing else. Native Android is out
of scope: no Kotlin, no APK, no Play Store listing
([ARCHITECTURE_RULINGS.md](ARCHITECTURE_RULINGS.md) R-007). This directory is
therefore no longer "the prototype" — it is the product. Python, GPUs and server
infrastructure remain, as training and evaluation infrastructure.

Specifications live in [`specs/`](../specs): product
([vision](../specs/01-product/PRODUCT_VISION.md),
[requirements](../specs/01-product/PRODUCT_REQUIREMENTS.md)), architecture
([system](../specs/02-architecture/SYSTEM_ARCHITECTURE.md),
[AI](../specs/02-architecture/AI_ARCHITECTURE.md),
[privacy](../specs/02-architecture/PRIVACY_ARCHITECTURE.md),
[learning](../specs/02-architecture/LEARNING_ARCHITECTURE.md)),
[plan](../specs/03-plan/IMPLEMENTATION_PLAN.md), web
([frontend audit](../specs/04-web/FRONTEND_AUDIT.md),
[inference strategy](../specs/04-web/WEB_INFERENCE_STRATEGY.md)), testing
([browser matrix](../specs/05-testing/BROWSER_MATRIX.md),
[definition of done](../specs/05-testing/DEFINITION_OF_DONE.md)).

## Android toolchain blocker — CLOSED, not fixed

The previous blocker (no JDK, no `JAVA_HOME`, no Android SDK) is closed because
the target changed, not because anything was installed. No further time goes
into Android tooling. `../Nexora` is left on disk, untouched and unused.

The required development environment is now Node.js, a package manager, a
browser, TypeScript tooling, and a Python environment for training and
evaluation. Of those, only Python is actually set up: there is no
`package.json`, no TypeScript and no build step in the repository today
([FRONTEND_AUDIT.md](../specs/04-web/FRONTEND_AUDIT.md)).

## This iteration's work

| § | task | state |
|---|---|---|
| 45.1–2 | Android removed from scope; blocker closed | **done** — R-007 |
| 45.3–11 | Product, architecture, privacy, learning, DoD, plan specs | **done** — `specs/` |
| 45.12 | Preserve AV-7 | **done** — `models/registry/assistive-av7-20260830/` |
| 45.13 | Preserve `av7_split` | **done** — `datasets/av7_split`, 545 MB, plus `av7_merged` |
| 45.14 | Correct validation/test terminology | **done** — R-008 |
| 45.15 | Design `assistive_test_v1` | **done** — designed, not built |
| 45.16 | Browser capability matrix | **done** — `specs/05-testing/BROWSER_MATRIX.md` |
| 45.17 | Audit the web frontend | **done** — KEEP / REFACTOR / DELETE per component |
| 45.18 | Decide what server/web code is retained | **done** — same audit |
| 45.19 | Best browser inference strategy | **partial** — three strategies researched, none chosen; the deciding measurement needs a phone |
| 45.20 | Benchmark the baseline in the web environment | **partial** — ONNX export measured on desktop; nothing measured in a browser |
| 10 | Model registry with hashes and provenance | **done** — `scripts/register_model.py` |
| 11 | Dataset manifest, hashes, recovery procedure | **done** — `scripts/dataset_manifest.py` |
| 13 | Video leakage policy | **done** — enforced by source recording, zero overlap measured |
| 18 | Dataset gap analysis | **done** — [DATASET_STATUS.md](DATASET_STATUS.md) |
| 30 | Browser-compatible model export | **done** — ONNX, faster than PyTorch |
| 19 | `stairs_down` stays experimental | **done** — R-005, unchanged |
| 20 | Outdoor capability | **not done** — still unmeasured |

## Perception

Two providers run per frame and are fused ([R-003](ARCHITECTURE_RULINGS.md)):

| provider | weights | vocabulary | authority |
|---|---|---|---|
| `general` | `yolov8n.pt` | COCO, 80 | vehicles, traffic light, dog, and `PERSON` jointly |
| `assistive` | `av_obstacle.pt` (AV-7) | 7 AV classes | stairs, door, dustbin, and `PERSON` jointly |

Measured on 336 **validation** images: fused recall **0.297**, against AV-7
alone 0.251 and COCO alone 0.232. Warm latency 79 ms for both providers against
~40 ms for one. Exported to ONNX, both run faster than PyTorch (AV-7: 27.6 ms vs
42.2 ms), at a cost of 21.6 MB gzipped to download both.

**Recall is low in absolute terms.** The best configuration finds under a third
of the annotated objects. That is the honest state of detection quality, it is
not improved by any of the architecture or documentation work in this iteration,
and it is the first priority.

## Announcement behaviour

The baseline, unchanged this iteration. Each criterion is pinned by a test in
`tests/test_realtime.cjs`:

1. An object briefly occluded is **not** re-announced as a new object.
2. A motionless object is announced **once**, however long it is observed.
3. A re-announcement waits out a 4 s floor unless the object became critical or
   closed to half its distance.
4. An approach is still reported — a chair closing from 10 steps announces at
   10, 5 and 2.
5. A persistent condition backs off, doubling to a 4-minute cap.

Detection quality and announcement quality are tracked separately. A quiet app
with poor recall is not an improved app.

## Known limitations

| limitation | evidence |
|---|---|
| Overall recall ≤ 0.30 | [MODEL_EVALUATION.md](MODEL_EVALUATION.md) |
| Vehicle and outdoor coverage asserted, never measured | no annotated outdoor set exists |
| No locked test set | `assistive_test_v1` designed, not built |
| `stairs_down` unsupported | 60 train boxes, 5 validation instances; R-005 |
| `table` near-unlearnable | 75 train instances; recall 0.059 |
| Walking surfaces, kerbs, ramps absent | need segmentation; no segmentation capability exists |
| Camera frames leave the device on every frame | [PRIVACY_ARCHITECTURE.md](../specs/02-architecture/PRIVACY_ARCHITECTURE.md) |
| API unauthenticated and unrate-limited | not acceptable for deployment (§38) |
| Production path runs through a developer laptop | Cloudflare quick tunnel; §37 forbids it |
| No PWA, no build, no workers, no TypeScript | [FRONTEND_AUDIT.md](../specs/04-web/FRONTEND_AUDIT.md) |
| Nothing measured on a phone | [BROWSER_MATRIX.md](../specs/05-testing/BROWSER_MATRIX.md) |
| Ultralytics AGPL-3.0 unresolved | R-006 |

## Next unblocked work, in order

1. **Improve perception.** Recall 0.297 gates everything. Start with the indoor
   part of `assistive_test_v1`, which can be carved from recordings already on
   disk, then outdoor capture, then a larger backbone — the export numbers show
   latency headroom at `yolov8n`'s 3M parameters.
2. **Outdoor evaluation set.** The one safety claim with no measurement.
3. **A deployed origin.** Blocks every browser benchmark, and §37 forbids the
   current laptop dependency.
4. **The web platform** — build, TypeScript, worker boundary, manifest, service
   worker — around the components the audit marked KEEP.
5. **Shadow-test harness.** Would have caught AV-6's vehicle blindness before
   deployment rather than after.
6. **`stairs_down` footage** shot from the top of a staircase. Nothing public
   supplies it.

Open tidying, not on the critical path: the per-class and fusion metrics live in
`scratchpad/evaluate.py` and belong in the existing `scripts/evaluate.py`; the
`training/` consolidation §29 describes is partly satisfied by
`models/registry/` and the manifest scripts.
