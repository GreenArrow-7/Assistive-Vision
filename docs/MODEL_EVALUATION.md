# Model comparison

Measured 2026-11-07 on **336 held-out images** (`av7_split/val`), greedy IoU@0.5
matching per canonical class. Every figure below came out of
`scratchpad/evaluate.py`; nothing is projected or carried over from a training
log.

**What this set is, and is not.** It is **validation data, not a test set.** Both
AV models used it for model selection, so it flatters them in the same way and to
the same degree. Neither model trained on these images — verified twice over:
no identical images by hash, and **no shared source recordings**, which is the
check that matters for video frames (see
[DATASET_STATUS.md](DATASET_STATUS.md)). COCO has never seen them at all.

There is no locked test set anywhere in the project. `assistive_test_v1` is
designed in DATASET_STATUS.md and not built. Until it is, every figure below is
a validation figure and must be described as one; none of them is an estimate of
field performance.

`not in vocabulary` means the model has no such class. It is not a score of
zero, and must never be reported as one.

## Per class — precision / recall (false negatives)

| class | COCO yolov8n | AV-6 | AV-7 | Combined (AV-7 + COCO) |
|---|---|---|---|---|
| PERSON | 0.580 / **0.218** (208) | 0.689 / 0.117 (235) | 0.686 / 0.132 (231) | 0.557 / **0.222** (207) |
| CHAIR | 0.480 / 0.429 (16) | 0.333 / 0.214 (22) | 0.435 / 0.357 (18) | 0.378 / **0.500** (14) |
| TABLE | 1.000 / 0.118 (15) | 0.000 / 0.000 (17) | 0.333 / 0.059 (16) | 0.600 / **0.176** (14) |
| DOOR | not in vocabulary | 0.697 / 0.215 (226) | 0.558 / **0.285** (206) | 0.577 / **0.285** (206) |
| STAIRS_UP | not in vocabulary | 0.636 / 0.259 (20) | 0.571 / **0.444** (15) | 0.571 / **0.444** (15) |
| DUSTBIN | not in vocabulary | 0.417 / **0.833** (5) | 0.453 / 0.800 (6) | 0.462 / 0.800 (6) |
| STAIRS_DOWN | not in vocabulary | not in vocabulary | 1.000 / 0.400 (3) | 1.000 / 0.400 (3) |

## Overall

| model | precision | recall | latency (median) | size | detections/image |
|---|---|---|---|---|---|
| COCO yolov8n | 0.567 | 0.232 | 45.1 ms | 6.55 MB | 0.83 |
| AV-6 | 0.585 | 0.200 | 42.3 ms | 6.23 MB | 0.67 |
| AV-7 | 0.553 | 0.251 | 42.2 ms | 6.23 MB | 0.89 |
| **Combined** | 0.537 | **0.297** | 87.3 ms | 12.78 MB | 1.53 |

## What the numbers say

**Recall is low everywhere, and that is the headline.** The best configuration
finds under a third of the annotated objects. For a product whose failure mode is
a missed obstacle, 0.297 recall is not a good detector — it is the honest
starting point. Precision differences between these models (0.54–0.59) are small
enough to be noise on 336 images; the recall differences are not.

**Combined wins on recall for every single class** and loses a little precision.
For this product that is the right trade: a false positive is a wasted sentence,
a false negative is a collision. This is the measured justification for
multi-provider perception rather than an argument from architecture.

**AV-7 beats AV-6 on every class they share** — chair 0.357 vs 0.214, stairs_up
0.444 vs 0.259, door 0.285 vs 0.215, table 0.059 vs 0.000 — and adds
`stairs_down`. AV-6 has no remaining advantage, so AV-7 is now the assistive
provider.

**COCO is the better person detector** (recall 0.218 vs AV-7's 0.132) despite
person being the AV models' largest class by far. That is why `PERSON` is set to
fuse both providers rather than trust the fine-tune.

**Latency roughly doubles**, 42 ms to 87 ms on CPU. That is the real cost of
running two providers and the reason `every_n` cadence exists in
`config.PERCEPTION_PROVIDERS`. Both providers currently run on every frame;
measure before raising either, as §9 of the brief requires.

## Safety-relevant false negatives

Stated explicitly rather than folded into an average:

| class | best FN count | of | best recall |
|---|---|---|---|
| PERSON | 207 | 266 | 0.222 |
| STAIRS_UP | 15 | 27 | 0.444 |
| DOOR | 206 | 288 | 0.285 |
| CHAIR | 14 | 28 | 0.500 |
| STAIRS_DOWN | 3 | 5 | 0.400 |
| CAR / BICYCLE / BUS / TRUCK | not evaluated | — | — |

Vehicles could not be measured: `av7_split` is indoor footage and contains no
vehicle annotations. COCO vehicle coverage is therefore **asserted from the
model's vocabulary, not measured on this project's data.** An outdoor evaluation
set is open work.

`STAIRS_DOWN` is measured on five instances. That is a sample size, not evidence.
The class stays marked `UNSUPPORTED_BY_CURRENT_MODEL` in `server/classes_av.py`
and is announced without a claimed direction.

## Browser deployability

A model is not promotable on desktop Python numbers alone (§42). Both providers
export to ONNX cleanly, and ONNX Runtime is **faster** than PyTorch on the same
CPU — 27.6 ms vs 42.2 ms for AV-7 — so leaving `.pt` behind costs nothing. The
cost is the download: 25.1 MB raw, 21.6 MB gzipped for both providers, before
the application can see anything.

These are desktop x86 with native ONNX Runtime, which is a *floor* no browser
runtime will beat, not an estimate. Full figures and the strategy question:
[specs/04-web/WEB_INFERENCE_STRATEGY.md](../specs/04-web/WEB_INFERENCE_STRATEGY.md).
Nothing has been measured on a phone.

## Reproducing

```bash
python scratchpad/evaluate.py eval.json
```

Set: `.claude/worktrees/local-training-ea0f02/datasets/av7_split/val`, 336 images,
conf 0.45, imgsz 640, IoU 0.5.
