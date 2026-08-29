# DATA_SOURCING.md — Build the AV-6 dataset from the internet (India + UAE)

You don't have to record every environment yourself. This guide assembles the
dataset from three streams, in priority order:

| Stream | What it gives | Effort |
|---|---|---|
| **A. Your own videos** | Exact deployment conditions (camera height, blur, lighting) | ~1 hr recording |
| **B. YouTube walkthroughs (India/UAE)** | Environment variety: malls, hospitals, colleges you can't visit | `fetch_videos.py`, zero recording |
| **C. Public labeled datasets** | Thousands of pre-labeled boxes for the common classes, no annotation | `pull_open_datasets.py` |

Target mix (rule of thumb): ≥30% stream A, ≤40% B, ≤30% C. A model trained
only on B+C will underperform at demo time — YouTube tours are stabilized,
well-lit, waist-height; your phone feed is not. Keep some A in every class.

## Stream B — YouTube walkthroughs

Curated starter manifest: `scripts/video_manifest.csv` (India: Lulu Mall
Trivandrum, AIIMS Delhi/Bhubaneswar/Jammu, LSR college; UAE: Dubai Mall × 3).
Add rows freely — good search patterns: `"walking tour" 4K <place> inside`,
`hospital corridor walkthrough <city>`, `campus tour inside classrooms <college>`.

```bash
pip install yt-dlp
python scripts/fetch_videos.py                    # all rows -> videos/
python scripts/fetch_videos.py --country uae --place mall
python scripts/build_dataset.py --videos videos/*.mp4 --out datasets/av_raw --fps 1
```

`build_dataset.py` already handles the rest: 1 fps extraction, blur/duplicate
dropping, YOLO pre-labels for import into Roboflow.

**Frame budget per video:** a 20-min tour at 1 fps ≈ 1,200 raw frames, maybe
500 after dedup. Cap what you upload — 300–400 *diverse* frames per video
beats 1,200 near-identical ones. In Roboflow, delete escalator/shopfront-only
frames fast; annotate frames containing stairs, doors, signs, dustbins, poles.

**Licensing (state this in the paper):** frames from standard-license YouTube
videos are used for academic training only; the raw frames are not
redistributed — only the URL manifest and trained weights are published
(same convention as Kinetics / YouTube-8M). `--cc-only` restricts to
Creative-Commons videos whose frames can be redistributed with attribution.

## Stream C — Public labeled datasets

### C1. Open Images V7 (auto-converted by script)

```bash
pip install fiftyone
python scripts/pull_open_datasets.py --per-class 400 --out datasets/oi_av14
```

Covers person, chair, table, door, stairs, dustbin with existing boxes.

**Must-do:** Open Images has ONE "Stairs" class, so every staircase arrives as
`stairs_up`. `stairs_down` is the sole critical class, so the descending ones
have to be split out:

```bash
python scripts/stairs_queue.py --make        # 579 boxes / 438 images queued
python scripts/review_crops.py --queue datasets/oi_av14/stairs_queue.csv \n    --images datasets/oi_av14/images         # 1 = up, 2 = down, D = drop
python scripts/stairs_queue.py --apply       # rewrites the label classes
```

This stays inside the by-name pipeline; a Roboflow round trip re-imports by
class INDEX, which is the collision `docs/TRAINING.md` Step 2 warns about.
To then TRAIN the class, move `stairs_down` from `AV_RETIRED` to `AV_CLASSES`
in `server/classes_av.py` and re-run `reindex_labels.py` + `prepare_split.py` —
until that edit the reindexer drops the new boxes, because the trained schema
has no such class.

### C2. Roboflow Universe (download in Roboflow, remap classes on import)

| Need | Source |
|---|---|
| stairs (6k images) | universe.roboflow.com/stairs-detection-6bozl |
| indoor objects for VI | universe.roboflow.com/benjie-mugagga-odw5x/detecting-indoor-objects-for-visually-impaired |
| exit signs | universe.roboflow.com/search?q=class:exit — pick a signage set |
| generic obstacles | universe.roboflow.com/visually-impaired-obstacle-detection-uxdze/obstacle-detection-yeuzf |

In Roboflow: "Clone/Import" into your AV project → it prompts for class
remapping → map to the 14 annotation-vocabulary names, drop classes you don't use.

### C3. MCIndoor20000 — doors, stairs, **hospital signs** (perfect fit)

github.com/bircatmcri/MCIndoor20000 — 20k images from a hospital: doors,
stairs, signs. Classification-labeled (no boxes): upload to Roboflow, keep the
class as a tag, draw the (usually single, obvious) box. Fast win for
`sign_*` classes, which are your scarcest.

### What no public set provides
`pole`, `sign_washroom/lift/reception/wheelchair` in Indian/UAE styling, and
`signboard` in Kannada/Hindi/Arabic. These MUST come from streams A/B —
prioritize them when annotating. `signboard` is the reason: 50 boxes from the
walkthrough frames trained to AP50 0.000, so the class is retired until the
annotation pass gives it real support.

## Merge + train (unchanged pipeline)

1. One Roboflow project, class list = the 14 names of `AV_ALL_CLASSES`
   (`server/classes_av.py`), same order; the trained AV-6 subset is `scripts/av6.yaml`.
2. Upload: A-frames + B-frames (with pre-labels) + `datasets/oi_av14` + Universe clones.
3. Correct/annotate per the rules in `docs/TRAINING.md` Step 2.
4. Generate version (70/20/10, brightness ±25%, blur ≤1px, rotation ±10°) → export YOLOv8.
5. Train exactly as `docs/TRAINING.md` Step 3 (Colab T4, yolov8s, imgsz=832).
6. Deploy: `models/av_obstacle.pt` → restart → `/health` shows `"object_schema": "av6"`.

For the paper, report per-class AP split by source mix (A only vs A+B+C) —
that ablation is itself a contribution.
