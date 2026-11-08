# Dataset and model-learning architecture

How data becomes a deployed detector, and what has to be true at each step before
it moves to the next. Current contents are in
[DATASET_STATUS.md](DATASET_STATUS.md); measured results are in
[MODEL_EVALUATION.md](MODEL_EVALUATION.md).

Training is **not** part of the safety runtime. Nothing in this pipeline runs on
a user's device or inside the server that answers frames. It is the project's
long-term learning capability and is kept deliberately separate from it.

## The pipeline

```
 footage / Open Images
        |
        v
   av_raw            auto-labels from a COCO model; COCO class ids, not ours
        |
        v
 annotate_upload     hand-off to the annotation pass (Roboflow / CVAT)
        |
        v
  av{N}_merged       human boxes merged with kept auto-labels
        |
        | scripts/reindex_labels.py   maps label indices BY NAME onto the
        |                             trained vocabulary; never by position
        v
  av{N}_split        train / val, with data.yaml  <- the only trainable artefact
        |
        v
  candidate model    models/av{N}_candidate.pt
        |
        v
   evaluation        held-out images, per-class recall, safety FN counts
        |
        v
   shadow test       runs beside the live provider, announces nothing
        |
        v
  safety review      recall on hazard classes; no silent class regressions
        |
        v
   promotion         becomes a provider in config.PERCEPTION_PROVIDERS
        |
        v
    rollback         previous candidate retained; swap the path back
```

## Rules that hold at each stage

**Vocabulary is data-driven.** `classes_av.AV_CLASSES` states only what the
weights can actually do. A class with too few boxes to learn is listed in
`AV_RETIRED`, not in the trained schema — a declared class that can never fire
reports 0 AP, drags macro mAP down, and tells every report a capability the
product does not have.

**Reindex by name, never by position.** Removing a class shifts every index above
it. Label files written against a 14-class vocabulary mean something different
under a 6-class one. `scripts/reindex_labels.py` maps by name for this reason.

**Splits are isolated by source recording, not by file hash.** This is stronger
than the rule it replaces, and it had to be: these images are video frames, and
frame 00699 and frame 00700 of one walkthrough have different hashes and nearly
identical content, so a hash check can report no leakage while a model is
validated on its own training data. One recording goes entirely to one split.
`scripts/dataset_manifest.py` groups by source and exits non-zero on a
violation, so it can gate the pipeline. `av7_split` passes: 1,157 training
sources, 252 validation sources, none shared
([ARCHITECTURE_RULINGS.md](ARCHITECTURE_RULINGS.md) R-008).

**A locked test set is read, at most, once per promotion decision.** Never for
model selection, threshold tuning, architecture choice, early stopping or
hyperparameters. `av7_split/val` is **not** one — both AV models selected on it,
so every figure derived from it is a validation figure. `assistive_test_v1` is
designed in [DATASET_STATUS.md](DATASET_STATUS.md) and not built.

**A registered model is identified by what is inside the file.**
`scripts/register_model.py` reads the class names, training date and config out
of the checkpoint, derives the registry key from them, and records both hashes,
the size, the dataset id and the licensing position. A weight file with no class
names inside it is refused rather than registered under a guess.

**A model is identified by its class names.** Never by its path or filename. Two
files named `av_obstacle_candidate.pt` differed by one class and 1,088 bytes, and
cost a day of searching — see
[ARCHITECTURE_RULINGS.md](ARCHITECTURE_RULINGS.md) R-002. `detector.detect_schema`
enforces this at load and refuses a vocabulary it cannot place.

**Promotion is a config edit, not a code edit.** A promoted model becomes an
entry in `config.PERCEPTION_PROVIDERS`. Rollback is restoring the previous path.
No model name appears anywhere else in the server.

**Evaluation precedes promotion, and says what it did not measure.** AV-6 was
deployed on one demo image before it was measured; the measurement later showed
AV-7 better on every class. Vehicle recall is still unmeasured because the
project owns no annotated outdoor set.

## Gaps in this pipeline today

| stage | state |
|---|---|
| raw → annotate | works; 986 frames through it |
| annotate → merged | works |
| merged → split | works, `scripts/reindex_labels.py` |
| split → candidate | works; AV-6 and AV-7 both produced this way |
| candidate → evaluation | **new**, `scratchpad/evaluate.py`; not yet a project script |
| evaluation → shadow test | **not built** |
| shadow → safety review | **not built**; no recall gate exists |
| **evaluation → browser benchmark** | **not built**; nothing measured on a phone |
| manifest and leakage gate | **works**, `scripts/dataset_manifest.py` |
| model registry | **works**, `scripts/register_model.py` |
| web export | **works**, ONNX; filed under the registry entry's `exports/` |
| promotion | manual file copy plus a config path |
| rollback | manual; previous candidates retained in `models/registry/` |
| locked test split | **does not exist** for any dataset |

Three gaps matter most. The missing **locked test set** means every number the
project reports is a validation number. The missing **shadow test** is what
would have caught AV-6's vehicle blindness before deployment rather than after.
And the missing **browser benchmark** means a model could pass every gate above
and still be unusable on the device the product runs on — which §42 of the
current brief forbids promoting on.
