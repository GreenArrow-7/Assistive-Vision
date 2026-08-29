# ANNOTATION_BRIEF.md — the one manual step, and how to not waste it

`datasets/annotate_upload.zip` (122 MB, 986 image+label pairs) is ready to
upload. Everything else in the pipeline is automated; this is the step that
cannot be.

## Why this exists

Six classes are trained. Eight names in the vocabulary have no usable boxes,
and the two that matter most are unreachable without hand-drawn boxes:

| name | boxes today | why no dataset supplies it |
|---|---|---|
| `stairs_down` | 0 | the **critical** class — the only one that fires "Warning! Stop and proceed carefully". Open Images photographs staircases from the bottom looking up, so it has none. |
| `signboard` | 50, AP50 0.000 | retired 2026-08-29 for lack of support. Text signs exist in the footage; the COCO pre-labeller only proposed one where it saw a "tv". |
| `pole` | 0 | absent from both the walkthrough footage and Open Images. |
| `sign_washroom`, `sign_exit`, `sign_lift`, `sign_reception`, `sign_wheelchair` | 0–4 | pictogram signs. This is the case OCR-only assistive readers fail on, and the reason the schema exists. |

An EasyOCR sweep was evaluated as a way to propose `signboard` boxes
automatically and rejected: 11 of 75 sampled frames carry legible text, 8 of
them from one mall video, and a text region is not a sign extent — such boxes
would supervise text detection that EasyOCR already performs. There is no
substitute for drawing them.

## Upload

1. roboflow.com → Create Project → Object Detection.
2. Upload **`images/` and `labels/` together** from the unzipped bundle, so the
   2,760 existing boxes import as pre-labels and you correct rather than draw.
3. Set the class list to **exactly the 14 names in `classes.txt`, in that
   order**. The `.txt` files carry integers, so a different order silently
   renames every box — `signboard` would import as `stairs_down`. Copy the file,
   do not retype it.

```
0 person        4 stairs_up     8 stairs_down    12 sign_reception
1 chair         5 dustbin       9 pole           13 sign_wheelchair
2 table         6 signboard    10 sign_washroom
3 door          7 —            11 sign_exit
```
(`classes.txt` is authoritative; the row above is a reading aid.)

## What to draw

Priority order — annotate in this order so a partial pass is still useful:

1. **`stairs_down`** — the steps descend away from you. Box the whole visible
   flight, not individual steps. Never merge with `stairs_up`. This class alone
   reactivates the critical-alert path.
2. **`signboard`** — any sign carrying text. It needs a few hundred boxes to
   beat the 0.000 AP it earned at 50.
3. **`sign_*`** — pictogram signs, boxed **even when they carry no text**. A
   washroom sign with text gets **both** a `sign_washroom` and a `signboard`
   box; overlapping boxes on different classes are correct and useful.
4. **`pole`** — free-standing vertical obstacles.

Rules that change the outcome:

* **No generic "obstacle" class.** It has no consistent appearance, annotators
  disagree, and mAP collapses. Hazard status is assigned in code
  (`AV_HAZARDS`), never by drawing.
* **Delete vehicle boxes** (car, bus, truck, motorcycle, bicycle, train) rather
  than remapping them. The schema has no vehicle class; those are handled by
  the COCO fallback model.
* **Frames with nothing in them are worth keeping** as background negatives —
  359 of the 986 are already empty on purpose.
* Correct the pre-labels as you go: the COCO pre-labeller calls dustbins
  "vase" and poles "parking meter".

## Coming back

4. Roboflow → Generate version → 70/20/10, brightness ±25%, blur ≤1 px,
   rotation ±10° → Export **YOLOv8**.
5. Un-retire whatever you annotated: move those names from `AV_RETIRED` to
   `AV_CLASSES` in `server/classes_av.py`. Only add a name that now has boxes on
   **both** sides of the split — `prepare_split.py` and the Colab trainer both
   refuse a class they cannot populate, which is the point.
6. Re-run the pipeline against the export:

```bash
python scripts/reindex_labels.py --src <roboflow_export> --out datasets/av_merged
python scripts/prepare_split.py --src datasets/av_merged --out datasets/av_split \
    --schema av<N> --extra datasets/oi_av<N>
```

`reindex_labels.py` maps **by name**, reading the export's own `classes.txt`,
so a Roboflow class reorder cannot corrupt the labels. That is the whole reason
the re-import goes through it rather than straight into training.

7. Retrain (`scripts/train_av14_colab.py`), then
   `python scripts/evaluate.py --compare runs/eval/coco_baseline.json runs/eval/<new>.json`.

## How much is enough

The harness treats any class under 30 val boxes as noise and prints the support
beside every metric. Aim for **≥200 train / ≥30 val boxes** per class you intend
to un-retire. Below that you get a number that cannot be defended at a review —
which is exactly how `signboard` came to be retired.
