# Architecture rulings

Decisions that are settled, with the evidence that settled them. A ruling is
reopened by new evidence, not by rediscovering the original question.

---

## R-001 — Do not merge `origin/main` to look for model weights

**Date** 2026-11-07 · **Status** settled

`origin/main` is 11 commits ahead and touches 2,937 files. It was proposed as a
way to obtain the AV-7 weights. It cannot be, and the merge was refused.

Evidence:

- `git ls-tree -r origin/main` lists **no `.pt` file at all**. `models/` is
  gitignored, so no weights have ever been in this repository.
- Five of the changed files collide with uncommitted work in the working tree
  (`.gitignore`, `server/config.py`, `server/main.py`,
  `tests/test_pipeline_logic.py`, `web/index.html`). Git refuses the merge
  rather than overwrite them, and clearing the way means committing or stashing
  21 files of unreviewed work.
- The AV-7 weights were subsequently found on local disk (R-002), so the merge
  had no remaining purpose.

A merge of `origin/main` needs its own justification — wanting the paper, or the
field-test fixes — and a clean working tree. Not this.

---

## R-002 — AV-7 exists; it was in a worktree, identified by class names

**Date** 2026-11-07 · **Status** settled

AV-7 was reported missing because every filename search returned the AV-6 file.
Both are named `av_obstacle_candidate.pt`. They differ by 1,088 bytes.

Found at
`D:\Youhan One\Assistive-Vision\.claude\worktrees\local-training-ea0f02\models\av_obstacle_candidate.pt`
and verified by loading the weights and reading `model.names`, never by path:

```
classes : person, chair, table, door, stairs_up, dustbin, stairs_down
trained : 2026-08-30T18:09:17, av7_split/data.yaml, 40 epochs, imgsz 640
md5     : e24b26075ae6…   size 6,227,242
```

Preserved as `models/av7_candidate.pt`. Nothing was moved or deleted.

**Rule carried forward:** identify weights by the class names inside the file.
The filename is not evidence, and twice now it has been actively misleading.

---

## R-003 — Perception is multi-provider; no single model is the detector

**Date** 2026-11-07 · **Status** settled

Replacing COCO with the AV fine-tune was trialled and is rejected as an end
state. The product must work on a road and in a corridor, and neither model
covers both: the AV models have never seen a car, COCO has never seen a
staircase or a door.

Measured on 336 held-out images ([MODEL_EVALUATION.md](MODEL_EVALUATION.md)),
running both providers and fusing gives the best recall of any configuration on
**every class**, for 0.03 precision and 45 ms. Recall is the metric that matters
when the failure mode is a missed obstacle.

Implementation: `server/perception.py` holds the provider list, the canonical
class vocabulary, the per-class authority policy and the fusion rule. Model names
appear in `config.PERCEPTION_PROVIDERS` and nowhere else.

---

## R-004 — AV-6 is superseded by AV-7

**Date** 2026-11-07 · **Status** settled

AV-7 beats AV-6 on every class the two share, and adds `stairs_down`. AV-6 has
no remaining advantage and is kept only as a rollback artefact.

AV-6 was never a production solution; it was an experiment that removed vehicle
coverage, which R-003 now makes unnecessary.

---

## R-005 — `STAIRS_DOWN` stays unsupported despite being detectable

**Date** 2026-11-07 · **Status** open, blocked on data

AV-7 carries the class and scores precision 1.000 / recall 0.400 — on **five**
validation instances, from 65 training boxes. That is a sample size, not
evidence.

The detection is surfaced, because a missed descending staircase is the worst
outcome in this product, but **without a claimed direction**: the spoken name is
"stairs, and I cannot reliably tell whether they go up or down". Wording is the
safety control here, and `classes_av.STAIRS_DOWN_VALIDATED` gates it.

Unblocks when footage shot from the top of a staircase exists. Open Images
photographs staircases from below, so this cannot be sourced publicly.

---

## R-006 — Ultralytics AGPL-3.0 is an unresolved release blocker

**Date** 2026-11-07 · **Status** open

Every model in the project derives from Ultralytics YOLOv8 weights, which are
AGPL-3.0. Distributing a product built on them carries obligations nobody has
decided on. Recorded here because it affects shipping, not accuracy, and has no
owner.
