# TRAINING.md — Fine-tune the detector on YOUR environments (Tier 1)

End-to-end: walkthrough videos → dataset → trained model → auto-deployed.
Total effort: ~1 day recording/annotating + one overnight Colab run. Free.

## Step 0 — Record (you're doing this)
3 × 20 min videos (college, hospital, mall), phone at chest height tilted
slightly down, walk slowly, pause at every sign/door/staircase/obstacle,
capture each thing from ~1 m / 3 m / 6 m. Include dim and backlit scenes.

**Descending stairs?** `scripts/stairs_queue.py --make` queues every Open
Images staircase for a one-keypress up/down verdict in `review_crops.py`; see
`docs/DATA_SOURCING.md` C1. This is the critical class.

**Can't record everything?** See `docs/DATA_SOURCING.md` — it sources extra
footage from YouTube walkthroughs of Indian/UAE malls/hospitals/colleges
(`scripts/fetch_videos.py`) and pre-labeled boxes from Open Images
(`scripts/pull_open_datasets.py`). Keep ≥30% self-recorded frames.

## Step 1 — Extract + pre-label frames (on your laptop)
```powershell
pip install ultralytics opencv-python
python scripts/build_dataset.py --videos college.mp4 hospital.mp4 mall.mp4 --out datasets/av_raw --fps 0.5
```
Output: `datasets/av_raw/images/` + `labels/` (YOLO format, pre-labeled by
yolov8s so you only correct, not draw). The existing 1,202 frames were cut at
0.5 fps with one running frame counter across all videos — re-running with the
same videos, order and fps regenerates the same file names, so the tracked
labels re-attach (images are gitignored; labels are tracked).

## Step 2 — Annotate in Roboflow (free tier)

**The bundle is already built: `datasets/annotate_upload.zip`** (986 image+
label pairs, 2,760 boxes, class list in canonical `AV_ALL_CLASSES` order).
See `docs/ANNOTATION_BRIEF.md` for what to draw, in what order, and how much
is enough. The steps below are the mechanics.
1. roboflow.com → Create Project → Object Detection.
2. Upload `images/` **and** `labels/` together (it imports the pre-labels).
3. Set the class list to **exactly these 14 names, in this order** — the
   annotation vocabulary `AV_ALL_CLASSES` in `server/classes_av.py`. The first
   seven are the trained AV-7 schema (`scripts/av7.yaml`); the last seven are
   annotation-only until they have boxes, and `scripts/reindex_labels.py`
   maps labels onto the trained schema by name:

   | # | class | # | class |
   |---|---|---|---|
   | 0 | person | 7 | signboard (annotation-only) |
   | 1 | chair | 8 | pole (annotation-only) |
   | 2 | table | 9 | sign_washroom (annotation-only) |
   | 3 | door | 10 | sign_exit (annotation-only) |
   | 4 | stairs_up | 11 | sign_lift (annotation-only) |
   | 5 | dustbin | 12 | sign_reception (annotation-only) |
   | 6 | stairs_down (**critical**; trained since the 2026-08-30 re-tag) | 13 | sign_wheelchair (annotation-only) |

   **Annotation rules — follow strictly or mAP suffers:**
   * No generic "obstacle" class. It has no consistent appearance; annotators
     disagree and the model can't learn it. Hazard status is assigned in code
     (`AV_HAZARDS`), not by drawing boxes.
   * `stairs_down` = you can see the steps descending away from you (the
     highest-risk class in the system — it triggers a "Warning! Stop and
     proceed carefully" alert). `stairs_up` = ascending. Never merge them.
   * Box the **whole visible flight** for stairs, not individual steps.
   * `signboard` = any text sign. It is currently annotation-only: 50 boxes
     trained to AP50 0.000, so it was retired from the schema until it has
     boxes (see `server/classes_av.py`). Reading text signs does not depend
     on it — OCR runs over the whole frame. The 5 `sign_*`
     classes = pictogram signs, boxed even when they carry no text — this is
     precisely the case OCR-only assistive readers fail on, and it makes your
     symbol recognition *trained* rather than keyword-mapped.
   * A washroom sign with text gets BOTH `sign_washroom` and `signboard`
     boxes (overlapping boxes on different classes are fine and useful).
   * Correct the pre-labels: the COCO pre-labeler will mislabel dustbins as
     "vase", poles as "parking meter" etc. Fix the class, keep the box —
     EXCEPT for vehicles (car, bus, truck, motorcycle, bicycle, train):
     DELETE those boxes entirely. AV-7 has no vehicle class, and the COCO
     indices collide with the annotation vocabulary (bus=5 imports as
     dustbin, car=2 as table), so "keeping the box" poisons the most safety-critical classes.
   * Do not upload raw pre-labels: run `python scripts/remap_to_av14.py`
     first. It converts the safe classes, deletes vehicle boxes, and queues
     the ambiguous ones to review_queue.csv; fold your decisions back with
     `python scripts/merge_review_queue.py`, then upload the merged set.
4. Generate version: 70/20/10 split, augmentations: brightness ±25%,
   blur ≤ 1 px, rotation ±10°. Export → **YOLOv8** → copy the download code.

## Step 3 — Train on Google Colab (free T4, overnight)
No-Roboflow path: `python scripts/prepare_split.py --src datasets/av7_merged
--out datasets/av7_split --schema av7 --extra datasets/oi_av7`, zip the split,
then run `scripts/train_av14_colab.py` in Colab (it refuses a split with an
empty class unless `ALLOW_SPARSE`). Roboflow path:

New notebook → Runtime → T4 GPU → run:
```python
!pip -q install ultralytics roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
ds = rf.workspace("YOUR_WS").project("YOUR_PROJECT").version(1).download("yolov8")

# imgsz=832: signs are small objects — 640 loses them at distance
!yolo detect train model=yolov8s.pt data={ds.location}/data.yaml \
    epochs=120 imgsz=832 batch=12 patience=30 \
    degrees=8 hsv_v=0.5 fliplr=0.5 mosaic=1.0 close_mosaic=15 \
    name=av7

# per-class AP -> paper Table T3. Watch stairs_down and the sign_* classes:
# they are the classes that justify the whole contribution.
!yolo detect val model=runs/detect/av7/weights/best.pt \
    data={ds.location}/data.yaml

from google.colab import files
files.download('runs/detect/av7/weights/best.pt')   # rename -> av_obstacle.pt
```

## Step 4 — Deploy (zero code changes)
Put the file at `models/av_obstacle.pt` and restart the server. The detector
auto-loads it (check `/health` → `"object_schema": "av7"`) and stops
filtering to COCO classes, so your new stairs/door/sign classes flow straight
through detection → steps → priority speech.

**The schema is read from the model's class names, not from the filename.**
If you drop a model there whose classes are COCO (e.g. you trained before
correcting the labels to AV-7), `/health` reports `"object_schema": "coco"`
and COCO hazard semantics are applied — it does *not* pretend to be AV-7.
This matters: `AV_HAZARDS` contains no vehicles, so a COCO model treated as
AV-7 would silently stop flagging cars, buses and bikes as hazards. A model
matching neither vocabulary is refused at startup and surfaces as
`/health` → `"error"`.

Heights for all 14 annotation-vocabulary names are already defined (`server/classes_av.py`
AV_HEIGHTS), spoken names are mapped ("sign_washroom" -> "washroom sign"),
hazard roles are assigned, and voice keywords route to the trained classes
("find washroom" -> class `sign_washroom`). Nothing else to configure.

## Step 5 — VFOV calibration (per phone, 2 minutes)
In the app: say **"calibrate"** (or tap 📐 Calibrate) → place a chair exactly
4 steps (~3 m) ahead → tap the screen. The app solves your phone's true field
of view from the measurement and persists it. Do once per device.

## Step 6 — Text detector (for the paper's oriented-text claim)
Same Colab, after registering at rrc.cvc.uab.es for ICDAR-2015:
```bash
python scripts/convert_icdar_to_obb.py --images train_images --gts train_gts --out datasets/icdar15_obb/train
python scripts/convert_icdar_to_obb.py --images test_images  --gts test_gts  --out datasets/icdar15_obb/val
yolo obb train model=yolov8n-obb.pt data=scripts/icdar15_obb.yaml epochs=80 imgsz=960 degrees=30
```
Drop `best.pt` at `models/text_obb.pt` → pipeline switches automatically.

## Expected outcome
COCO-generic yolov8n → your yolov8s fine-tune typically moves in-environment
precision from ~60–70% to 90%+, kills the false positives at the source, and
adds the stairs/door/sign classes a blind user actually needs. Run `yolo val`
before/after and put both rows in the paper.
