# Learning architecture

Two loops that never touch each other.

## Fast loop — personal, in the browser

The user teaches the app about their own world. Custom labels, remembered
places, corrections to what it said, preferences about verbosity and units.
Stored in IndexedDB on the device, viewable and deletable by the user
([PRIVACY_ARCHITECTURE.md](PRIVACY_ARCHITECTURE.md)).

This loop changes **what the app says about a detection**. It never changes the
detector.

## Slow loop — central, offline, gated

```
dataset -> training -> candidate model -> validation -> LOCKED TEST
        -> browser benchmark -> safety review -> promotion -> registry
                                                     |
                                                  rollback
```

The pipeline, its rules and its current gaps are in
[../../docs/DATASET_ARCHITECTURE.md](../../docs/DATASET_ARCHITECTURE.md). Two
gates on it are new requirements from this brief and are **not yet built**:

**The locked test set.** `assistive_test_v1`, designed in
[../../docs/DATASET_STATUS.md](../../docs/DATASET_STATUS.md). The training agent
may not use it for model selection, threshold tuning, architecture choice, early
stopping or hyperparameters. Every metric the project has reported so far is a
validation metric, and must be described as one until this exists.

**The browser benchmark.** A model that is good in desktop Python but too slow
in Chrome on a phone is not promotable (§42). Measured: latency, memory,
stability, thermal behaviour, frame rate, model download size. None of it has
been measured on a phone.

## Why they are kept apart

A user correcting "that's my office door, not a doorway" is teaching the
application about their flat. If that fed the detector, one person's building
would reshape everyone's obstacle detection, and a safety model would change
without passing any gate. So:

**Nothing a user does in a session retrains or replaces the safety detector, and
no training run swaps a detector inside a live session.** Promotion is a
deliberate configuration change with evaluation, a browser benchmark and a
safety review behind it, and a rollback path in front of it.

## Model learning lab layout

`scripts/` and `datasets/` carry this today. The directory shape §29 asks for —
`training/{datasets,registry,research,experiments,pipelines,evaluation,exports,reports}`
— is the intended consolidation; what exists now is `models/registry/`
(`scripts/register_model.py`), `datasets/*/MANIFEST.json`
(`scripts/dataset_manifest.py`) and the existing training and evaluation
scripts. Moving the rest is tidying, not capability, and is not on the critical
path while recall is 0.297.

Training uses Python and GPUs. That does not contradict the web-only product
requirement: this is engineering infrastructure, and the end-user application
remains a browser application (R-007).
