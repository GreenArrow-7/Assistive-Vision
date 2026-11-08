# Dataset status

Audited 2026-11-07 by script, not by eye: every number below was counted from the
files on disk (`scratchpad/audit_datasets.py`, image hashes for duplicate and
leakage checks). Nothing was deleted or moved.

## Where the data actually is

The project exists in three near-identical copies. The working copy is
`D:\Youhan One\Youhan One\Assistive-Vision`; the other two are
`D:\Youhan One\Assistive-Vision` and
`C:\Users\Google\Videos\Youhan One\Youhan One\Assistive-Vision`.

**No image data is in git.** `datasets/` and `models/` are ignored, so a clone
carries code only. This is why a search of `origin/main` for model weights found
nothing — see [ARCHITECTURE_RULINGS.md](ARCHITECTURE_RULINGS.md) R-001.

The AV-7 material lives only inside old Claude worktrees:

```
.claude/worktrees/local-training-ea0f02/datasets/av7_split
.claude/worktrees/admiring-keller-07a67c/datasets/av7_split
.claude/worktrees/festive-lichterman-ce977c/datasets/av7_split
```

That is a single point of failure for the best model the project has. Promoting
`av7_split` out of a worktree into `datasets/` is open work.

## Inventory

| dataset | images | boxes | splits | class names | state |
|---|---|---|---|---|---|
| `av6_split` | 2,070 | 5,428 | train 1,726 / val 344 | from `data.yaml` | **trainable** — AV-6 was trained here |
| `av7_split` *(worktree)* | 2,021 | 5,339 | train 1,685 / val 336 | 7, incl. `stairs_down` | **trainable** — AV-7 was trained here |
| `av6_merged` | 986 | 2,735 | unsplit | none | intermediate |
| `av_raw` | 986 | 3,770 | unsplit | none (COCO ids to 77) | raw auto-labels |
| `annotate_upload` | 986 | 2,760 | unsplit | none | annotation hand-off |
| `av14_merged` | 0 | 2,790 | unsplit | none | labels only |
| `av14_seed` | 0 | 2,723 | unsplit | none | labels only |
| `oi_av6` | 1,451 | 2,722 | unsplit | none | Open Images pull |
| `oi_av14` | 1,451 | 2,722 | unsplit | none | Open Images pull |
| `cocotext_obb` | 300 | 1,359 | train 240 / val 60 | 1 (`text`) | local subset; the real 13,880/3,261 set was built in Colab and died with the VM |

## Class balance, `av7_split`

| class | instances |
|---|---|
| person | 2,799 |
| door | 892 |
| dustbin | 760 |
| stairs_up | 426 |
| chair | 305 |
| table | 92 |
| **stairs_down** | **65** |

`person` outnumbers `stairs_down` 43 to 1. No class is empty, but `table` and
`stairs_down` are too thin to learn reliably, and the measured per-class recall
in [MODEL_EVALUATION.md](MODEL_EVALUATION.md) shows exactly that.

## Problems found

| dataset | problem | count |
|---|---|---|
| `av14_merged` | labels with no image | 1,202 |
| `av14_seed` | labels with no image | 1,161 |
| `av6_merged` | labels with no image | 216 |
| `av_raw` | labels with no image | 215 |
| `av6_merged` | empty label files | 560 |
| `av14_merged` | empty label files | 551 |
| `oi_av14` | malformed label rows | 438 |
| `annotate_upload` | empty label files | 359 |

Empty label files are legitimate in YOLO — they mark a true negative frame. They
are listed because a large count can also mean an annotation pass silently
produced nothing, and the two cases are indistinguishable without review.

**Duplicates and leakage: none in the split datasets.** Checked by MD5 over image
content:

```
av7_split val inside av7_split train : 0
av6_split val inside av6_split train : 0
av7_split val inside av6_split train : 0      <- AV-6 never saw the eval set
av7_split val identical to av6_split val : 336 of 336
```

The last line is what makes a fair comparison possible: AV-6 and AV-7 share the
same validation images, and neither trained on them.

## Licences and provenance

| source | provenance | licence |
|---|---|---|
| `oi_av6`, `oi_av14` | Google Open Images v7 subset | CC BY 4.0 (images), annotations CC BY 4.0 — attribution required in any publication |
| `av_raw`, `annotate_upload`, walkthrough frames | filmed by the project team | project-owned |
| `cocotext_obb` | COCO-Text v2 over COCO 2014 images | COCO images CC BY 4.0; COCO-Text annotations research use |
| `yolov8n.pt`, `yolov8s.pt`, `yolov8n-obb.pt` | Ultralytics pretrained | **AGPL-3.0** — this is a licence obligation on any distributed product built on these weights and needs a decision before release |

The Ultralytics AGPL position is unresolved and is a release blocker, not a
dataset issue. Recorded here because it is the only place the model lineage is
written down.

## Model-to-dataset relationships

| model | trained on | classes | md5 (first 12) |
|---|---|---|---|
| `av6_640_candidate.pt` = `av_obstacle_candidate.pt` | `av6_split` | 6 | `c40217621a15` |
| `av7_candidate.pt` | `av7_split` | 7 (+`stairs_down`) | `e24b26075ae6` |
| `text_obb.pt` | COCO-Text v2 (Colab) | 1 (`text`) | — |
| `yolov8n.pt` | COCO | 80 | — |

Both AV models were trained for 40 epochs at imgsz 640.

## Open work

1. `av7_split` and the AV-7 weights exist only inside a worktree. Promote them.
2. No test split anywhere — every dataset has train/val only. The numbers in
   MODEL_EVALUATION.md are therefore validation numbers, with the optimism that
   implies, and are labelled as such.
3. `stairs_down` needs footage shot from the top of a staircase. Open Images
   photographs staircases from below, so this cannot be sourced publicly.
4. `table` at 92 instances is close to unlearnable; measured recall is 0.06–0.18.
