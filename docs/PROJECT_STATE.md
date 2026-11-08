# Project state

Updated 2026-11-07. Claims here are measured or cite the file that backs them.
The web/server system is the **prototype and model-evaluation environment**; the
production target remains native Android (`../Nexora`).

## Iteration completion criteria

| # | Criterion | State |
|---|---|---|
| 1 | Speech-repeat fixes remain green | **met** — 230 Python, 4 Node suites |
| 2 | AV-6 classified as experimental | **met** — superseded by AV-7, rulings R-004 |
| 3 | Multi-provider architecture specified | **met** — `server/perception.py` |
| 4 | COCO + AV coexist in the prototype | **met** — both live, verified through the public link |
| 5 | Canonical class mapping exists | **met** — `perception.canonical()` |
| 6 | Duplicate person detections fused | **met** — `perception.fuse()`, `tests/test_perception.py` |
| 7 | Dataset readiness audit produced | **met** — [DATASET_STATUS.md](DATASET_STATUS.md) |
| 8 | Evaluation dataset identified | **met** — `av7_split/val`, 336 images, no leakage |
| 9 | Baseline model comparison produced | **met** — [MODEL_EVALUATION.md](MODEL_EVALUATION.md) |
| 10 | AV-7 search documented | **met** — found; rulings R-002 |
| 11 | Stairs-down limitation explicit | **met** — `STAIRS_DOWN_VALIDATED = False`, rulings R-005 |
| 12 | Android toolchain rechecked | **met** — still blocked, see below |
| 13 | PROJECT_STATE updated | this file |

## Perception

Two providers run per frame and are fused:

| provider | weights | vocabulary | authority |
|---|---|---|---|
| `general` | `yolov8n.pt` | COCO, 80 | vehicles, traffic light, dog, and `PERSON` jointly |
| `assistive` | `av_obstacle.pt` (AV-7) | 7 AV classes | stairs, door, dustbin, and `PERSON` jointly |

Measured: fused recall **0.297** against AV-7 alone 0.251 and COCO alone 0.232,
on 336 held-out images. Warm latency **79 ms** for both providers against ~40 ms
for one. Cadence (`every_n`) is configured per provider and both currently run on
every frame — measure before raising either.

**Recall is low in absolute terms.** The best configuration finds under a third
of annotated objects. That is the honest state of detection quality and is not
improved by the architecture work; the architecture stops the system having to
choose which environment to fail in.

## Announcement behaviour

Fixed this iteration and now the baseline. Acceptance criteria, each pinned by a
test in `tests/test_realtime.cjs`:

1. An object briefly occluded is **not** re-announced as a new object. Track
   lifetime (6.5 s) owns deletion, not the missed-frame counter.
2. A motionless object is announced **once**, however long it is observed.
3. A re-announcement waits out a 4 s floor unless the object became critical or
   closed to half its distance.
4. An approach is still reported — a chair closing from 10 steps announces at
   10, 5 and 2.
5. A persistent condition backs off, doubling to a 4-minute cap, instead of
   repeating every 30 s forever.

Detection quality and announcement quality are tracked separately. A quiet app
with poor recall is not an improved app.

## Known limitations

| limitation | evidence |
|---|---|
| Overall recall ≤ 0.30 | [MODEL_EVALUATION.md](MODEL_EVALUATION.md) |
| Vehicle coverage asserted, never measured | no annotated outdoor set exists |
| `stairs_down` unsupported | 65 training boxes, 5 validation instances |
| `table` near-unlearnable | 92 instances; recall 0.06–0.18 |
| No test split anywhere | every figure is a validation figure |
| AV-7 weights and `av7_split` live only in a worktree | [DATASET_STATUS.md](DATASET_STATUS.md) |
| Ultralytics AGPL-3.0 unresolved | rulings R-006 |

## Android toolchain — still blocked

Rechecked 2026-11-07:

```
java          : not found
JAVA_HOME     : unset
ANDROID_HOME  : unset
ANDROID_SDK_ROOT : unset
~/AppData/Local/Android/Sdk : absent
```

`../Nexora` has its Gradle wrapper, `app/`, and `build.gradle.kts`. Nothing can
be compiled or tested until a JDK and the Android SDK are installed. `:core`
compilation, `:core` tests and the debug build are all blocked on this, and it
stays the highest-priority environment blocker. Per the brief, no further
uncompiled Kotlin should be generated while this holds.

## Next unblocked work, in order

1. **Promote `av7_split` and the AV-7 weights out of the worktree.** The best
   model and its data have one copy, inside a temporary directory.
2. **Carve a test split** from existing data so figures stop being validation
   figures.
3. **An outdoor evaluation set.** Vehicle recall is the one safety claim with no
   measurement behind it.
4. **Shadow-test harness.** Would have caught AV-6's vehicle blindness before
   deployment rather than after.
5. **`stairs_down` footage** shot from the top of a staircase. Nothing public
   supplies it.

`scripts/evaluate.py` already exists in this project and was not used for the
comparison above — the harness in `scratchpad/evaluate.py` was written fresh.
Folding the per-class and fusion metrics into the existing script is open tidying.
