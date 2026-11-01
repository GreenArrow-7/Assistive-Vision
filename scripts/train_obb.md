# Training the YOLOv8-OBB text detector

Three recipes. A (ICDAR only) is the fast baseline; B is the 3-stage curriculum
(SynthText → COCO-Text → ICDAR-15) that papers use for the best oriented-text
results; C needs no registration and can be started today. All produce one file:
`models/text_obb.pt` (class 0 = text) — the pipeline auto-switches when it
exists (`/health` → `"text_obb": true`).

## Recipe A — ICDAR-2015 only (~1 evening)

1. Download ICDAR-2015 Incidental Scene Text (register at rrc.cvc.uab.es).
2. ```bash
   python scripts/convert_icdar_to_obb.py --images train_images --gts train_gts --out datasets/icdar15_obb/train
   python scripts/convert_icdar_to_obb.py --images test_images  --gts test_gts  --out datasets/icdar15_obb/val
   yolo obb train model=yolov8n-obb.pt data=scripts/icdar15_obb.yaml epochs=80 imgsz=960 degrees=30
   ```
3. `runs/obb/train/weights/best.pt` → `models/text_obb.pt`.

## Recipe C — COCO-Text only, no registration (~1 evening)

ICDAR-2015 needs an account; COCO-Text and the COCO images do not, so this is the
recipe for when registration is the thing in the way. COCO-Text annotates a subset
of COCO train2014 with polygon masks, which `convert_cocotext_to_obb.py` turns into
oriented boxes (`cv2.minAreaRect`), and it carries its own train/val split in the
`set` field, so no second dataset is needed to validate on.

1. Both downloads are open. Run them inside Colab, not over a home link:
   ```bash
   wget https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip   # 12 MB
   unzip -q cocotext.v2.zip
   wget http://images.cocodataset.org/zips/train2014.zip                          # 12.6 GB
   unzip -q train2014.zip
   ```
   Checked 2026-11-01: both answered 200. Only the ~20k annotated images are copied
   into the dataset, so the converter's output is about 4 GB.
2. ```bash
   python scripts/convert_cocotext_to_obb.py --json cocotext.v2.json --images train2014 --out datasets/cocotext_obb
   yolo obb train model=yolov8n-obb.pt data=scripts/cocotext_obb.yaml epochs=60 imgsz=960 degrees=30
   ```
   Fewer epochs than Recipe A on purpose: this set is about twenty times larger.
   The yaml follows the same convention as `icdar15_obb.yaml`.
3. `runs/obb/train/weights/best.pt` → `models/text_obb.pt`.

What it costs: COCO-Text is mostly small incidental text in everyday photographs, so
the detector is weaker on the blurry, strongly rotated signboards that ICDAR-15 is
made of — the deployment case. Treat it as a working first model. Recipe A or B
later resumes from these weights and only improves them, because the class list
(0 = text) never changes.

## Recipe B — SynthText → COCO-Text → ICDAR curriculum (2–3 Colab sessions)

Why staged: SynthText (synthetic, unlimited, perfect oriented labels) teaches
"what text looks like"; COCO-Text adds real photos / clutter / small text;
ICDAR-15 finishes on exactly the deployment case — incidental, blurry,
multi-oriented text from a moving camera. Same class (0=text) throughout, so
each stage simply resumes from the previous stage's weights.

### Stage 1 — SynthText pretrain (subset is enough)
```bash
python scripts/convert_synthtext_to_obb.py --root data/SynthText --out datasets/synthtext_obb --max-images 50000
# yaml: train=synthtext_obb/train/images, val=icdar15_obb/val/images (real val!)
yolo obb train model=yolov8n-obb.pt data=synthtext.yaml epochs=15 imgsz=800 mosaic=0.5 name=stage1
```
Validate on REAL data (ICDAR val), never on SynthText — synthetic val scores lie.

### Stage 2 — COCO-Text (real, complex scenes)
```bash
python scripts/convert_cocotext_to_obb.py --json cocotext.v2.json --images train2014 --out datasets/cocotext_obb
yolo obb train model=runs/obb/stage1/weights/best.pt data=cocotext.yaml epochs=40 imgsz=960 name=stage2
```

### Stage 3 — ICDAR-2015 fine-tune (deployment domain)
```bash
yolo obb train model=runs/obb/stage2/weights/best.pt data=scripts/icdar15_obb.yaml \
    epochs=60 imgsz=960 degrees=30 lr0=0.002 name=stage3   # lower LR: fine-tune
```
→ copy `best.pt` to `models/text_obb.pt`. For the paper, val each stage's
weights on ICDAR-15 test — the 3-row table (stage vs precision/recall/hmean)
demonstrates the curriculum's contribution.

### Colab disk/time notes
* SynthText full = 41 GB / 858k images. 50k converted ≈ 2.5 GB; Stage 1 ≈ 4 h
  on a free T4. Get it via academictorrents or the official VGG link.
* COCO-Text needs COCO train2014 (13 GB), but only ~20k images carry legible
  text — the converter copies only annotated ones (~4 GB output).
* Free Colab disconnects ≈ 12 h: run one stage per session; each stage resumes
  from the previous `best.pt`, so nothing is lost.

## Worth adding for India/UAE deployments
**ICDAR-2019 MLT** (rrc.cvc.uab.es/?ch=15) contains **Arabic, Hindi
(Devanagari) and Bangla** scene text — matches UAE/India signboards, which
SynthText / COCO-Text / ICDAR-15 (English-only) do not cover. Its GT is the
same `x1,y1,…,x4,y4,script,transcription` quad format — reuse
`convert_icdar_to_obb.py` unchanged (the extra "script" column joins into the
transcription field and is ignored). Insert as a Stage 2.5 or mix into
Stage 3. Detection will then find Arabic/Hindi words; to also READ them, set
EasyOCR languages to `['en','hi']` or `['en','ar']` in `text_pipeline.py`.
