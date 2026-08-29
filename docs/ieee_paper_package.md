# IEEE Publication Package — Assistive Vision System

**Working title:** *Priority-Aware Assistive Vision: Oriented Text Detection, Metric
Step-Distance Estimation, and Hands-Free Voice Interaction for Visually
Impaired Navigation*

**Suggested venues:** IEEE Access · IEEE Sensors Journal · IEEE Trans. on
Human-Machine Systems · (conference route: IEEE ICASSP / EMBC / ICCE).

**Integrity note (read first):** every result table below is a template. Run the
experiments, insert your measured numbers, and keep the training logs/weights —
reviewers increasingly ask for them. Never submit literature numbers as your own.

---

## 1. Abstract (draft — 200 words)

Visually impaired individuals depend on environmental text and spatial awareness
that conventional assistive readers cannot provide: signage is frequently rotated
or perspective-distorted, obstacle information is absent, and audio feedback is
unprioritized. We present an end-to-end assistive vision system combining (i) an
oriented-text pipeline that rectifies each detected quadrilateral to horizontal
before recognition, (ii) an obstacle detector fine-tuned on a six-class indoor
navigation schema, (iii) monocular distance estimation converted to *walking
steps* by fusing a class-height pinhole model with a ground-plane model and
taking the conservative minimum, (iv) a rule-based priority scheduler that orders
speech as hazards → user query → symbols → environment summary, and (v) a
hands-free voice interface covering the full application lifecycle. We further
contribute a *safety-by-construction* mechanism: the runtime identifies a model
by its class names rather than its filename and refuses an unrecognised
vocabulary, and the dataset tooling refuses a split declaring a class it cannot
populate — together closing a silent capability-loss failure mode that no
accuracy metric reveals. On our held-out split a preliminary
detector (YOLOv8n, 640 px, 12 epochs) achieves mAP@50 0.293 and raises
frame-level hazard recall from 0.236 to 0.527 at precision 0.987 over the
COCO baseline; per-frame server latency is 3,330 ms at p50 on a laptop CPU,
of which text recognition is 3,203 ms and detection 102 ms. The final
detector (YOLOv8s, 832 px, 120 epochs) is **[PENDING]**.
Code and the dataset pipeline are released.

> **Integrity.** Every `[PENDING]` is produced by `scripts/evaluate.py` on our own
> data. None is filled from the literature. Claims removed for lack of supporting
> work — a user study, a fine-tuned CRNN recogniser, MSRA-TD500 training and a
> released symbol dataset — are documented in `docs/paper_claims_audit.md`.

## 2. Contributions (the reviewer checklist)

* **C1 — Oriented-text pipeline with perspective rectification.** Each detected
  quadrilateral is warped to horizontal before recognition. *Status: the
  rectification and recognition stages are implemented and tested; the detector
  is currently CRAFT. A YOLOv8-OBB detector fine-tuned on ICDAR-2015 is in
  progress — until those weights exist this contribution must be stated as
  pipeline design, not as a trained detector.*
* **C2 — Step-metric spatial guidance**: two independent monocular estimators
  (class-height pinhole, ground-plane with gyroscope pitch correction) fused by
  taking the minimum, because under-estimating distance is the safe error for a
  blind user. Reported in walking steps, the unit the user can act on.
* **C3 — Priority-aware speech scheduler**: hazards → query → symbols →
  summary, with distant hazard-class objects demoted so only close ones
  interrupt.
* **C4 — Full-lifecycle voice interaction** (start/stop/find/describe/navigate/
  how-far/repeat/mute/calibrate/help) with self-echo suppression.
* **C5 — Reproducible dataset pipeline**: extraction, deduplication,
  pre-labelling, vocabulary remapping with an adjudication queue, by-name label
  reindexing, and a leakage-safe by-video split — 143 automated tests.
* **C6 — Safety-by-construction schema verification.** Models are identified by
  class names, not filenames; unrecognised vocabularies are refused at startup;
  and a split that declares a class it cannot populate is refused before
  training. This targets silent capability loss, the failure mode that matters
  when the user cannot see that the system stopped warning them.

## 3. Related work positioning (1 paragraph each)

EAST (Zhou et al., CVPR 2017) and MOST (He et al., CVPR 2021) address oriented
text detection but stop at detection; DBNet (Liao et al., AAAI 2020) and CRAFT
(Baek et al., CVPR 2019) similarly target benchmarks, not assistive delivery.
Assistive OCR systems (Google Vision-based readers; VI-OCR, 2026) provide
reading but no obstacle awareness, no metric distance, and no output
prioritization. Our gap: *the integration layer* — oriented detection feeding a
prioritized, step-metric, voice-interactive assistive loop — plus trained
models with released weights.

## 4. System architecture

Five layers (Input → Processing → AI → Logic → Output), client-server:
smartphone browser captures frames/voice, FastAPI server runs inference,
priority engine composes speech, phone renders TTS. Figure = repo README
diagram. Latency budget table (measured 2026-08-29, laptop CPU, preliminary AV-6
model): capture 30 ms · upload 60–150 ms · detection 102 ms p50 (153 p90) ·
OCR 3,203 ms p50 (3,864 p90) · logic <1 ms · TTS start ~100 ms. OCR is the
dominant cost and the obvious target for the next optimisation.

## 5. Methodology

### 5.1 Oriented text detection — YOLOv8n-OBB (trained)

* **Formulation:** anchor-free head predicting (x, y, w, h, θ), θ ∈ [−90°, 90°)
  long-edge definition; losses: BCE (cls) + DFL (box distribution) + **ProbIoU**
  rotated-box regression (Ultralytics 8.x default for OBB).
* **Two-stage training (from scratch → domain):**
  1. *Synthetic pre-train:* SynthText (858k images) converted to OBB via word
     quads; 3 epochs, imgsz 640, batch 64 (grad-accum if VRAM-limited),
     SGD lr₀ 0.01 cosine, mosaic 1.0.
  2. *Real fine-tune:* ICDAR-2015 (1,000 train); 80 epochs, imgsz 960,
     batch 16, lr₀ 0.005, **degrees=30 rotation aug** (critical for
     orientation generalization), mosaic off last 10 epochs, EMA on.
     MSRA-TD500 was considered and dropped: no conversion tooling was written
     for it, so claiming it would describe work not performed.

  *Status: PLANNED, not yet run.* Both converters are implemented and covered by
  tests against fixtures shaped like the real ground truth, but no OBB weights
  exist. Until they do, the deployed text detector is CRAFT and the paper must
  say so.
* **Conversion tooling:** `scripts/convert_icdar_to_obb.py` (released).
* **Rectification:** each predicted quad is perspective-warped to horizontal
  (`server/text_pipeline._deskew`) before recognition — report ablation.

### 5.2 Text recognition — stock EasyOCR (CRNN)

Recognition uses EasyOCR's English recogniser unmodified: ResNet features +
BiLSTM + CTC (CRNN, Shi et al., TPAMI 2017). **No recogniser fine-tuning was
performed**, and none is claimed. The contribution at this stage is the
rectification that precedes recognition, not the recogniser itself.

Fine-tuning it via deep-text-recognition-benchmark on SynthText and ICDAR-2015
crops, then exporting to EasyOCR's custom-model format, remains available as
future work. It is listed here as a direction, not a result.

### 5.3 Object & obstacle detection — AV-6

COCO covers person/vehicle/furniture but misses **stairs, doors and dustbins**
— classes an indoor assistive user needs. The schema is *AV-6*:

* `person, chair, table, door, stairs_up, dustbin`.
* Sources: 1,202 frames extracted at 0.5 fps from 8 indoor walkthrough videos
  (malls, hospitals, campuses; two public tours, six self-recorded), blur- and
  duplicate-filtered and pre-labelled by YOLOv8s so annotation is correction
  rather than drawing (3,770 pre-label boxes, 2,723 auto-remapped, 219
  hand-adjudicated); plus 1,451 ready-labelled Open Images V7 images (2,722
  boxes) supplying door/stairs/dustbin volume.
* Split **by whole video**, never by frame: consecutive 0.5 fps frames are
  near-identical, so a frame-level split leaks validation into training. The
  harness re-proves disjointness on every run. Open Images contributes its own
  upstream train/validation assignment, which is honoured rather than re-split.
* Resulting split (rebuilt 2026-08-29, pre-annotation): 1,726 train / 344 val
  frames; 4,753 train / 675 val boxes; 367 empty pre-label frames dropped. Four
  self-recorded videos are held out for validation. One self-recorded video
  (215 labelled frames) is not yet re-imported; the figures must be
  regenerated after annotation.

**Why six classes and not fourteen.** The schema was designed with fourteen.
Eight lack the boxes to learn, and a declared-but-unlearnable class is not free: the
model carries an output that can never fire, reports zero AP, and drags macro
mAP down — while the runtime's name-based check still certifies the weights as
ours. Declaring only what the weights can do is what makes C6 coherent. The
retired names are kept as the *annotation* vocabulary, so one box un-retires a
class. `stairs_down` is the significant omission: it is the sole critical class,
and public sources do not supply it — Open Images photographs staircases from
the bottom looking up. Acquiring descending-viewpoint footage is stated future
work, and the critical-alert path is documented as dormant until then.

### 5.4 Symbol recognition — keyword-routed, not yet trained

Symbols resolve two ways today: a detected class implies a symbol, and OCR text
is matched to a symbol vocabulary by **whole-word** match (substring matching
announced a restaurant *MENU* board as a washroom, because "men" is a substring
of both MENU and WOMEN; the regression is pinned by test).

The five pictogram `sign_*` classes are defined in the schema and routed by the
voice keyword map, but hold **zero annotated boxes**, so no symbol model is
trained and none is claimed. Training them is the highest-value remaining work,
because a pictogram with no text is precisely the case OCR-only assistive
readers fail on — but it requires annotation that does not yet exist.

## 6. Datasets summary

| Dataset | Size | Role | Status |
|---|---|---|---|
| ICDAR-2015 | 1,000/500 | OBB detector fine-tune + text eval | Converter ready; **not yet downloaded** (RRC registration) |
| SynthText | 858k synth | Optional OBB pre-train | Converter ready; **optional, not run** |
| Open Images V7 | 1,451 imgs / 2,722 boxes | door, stairs, dustbin volume | **Acquired** (`pull_open_datasets.py --per-class 400`) |
| Walkthrough video (ours) | 1,202 frames / 8 videos | Indoor domain frames | **Acquired**, 219 ambiguous boxes hand-adjudicated |
| **AV-6 split** (ours) | 1,726 train / 344 val frames; 5,428 boxes | Obstacle detector train + eval | **Built** 2026-08-29, leakage-safe by video; pre-annotation |

Classes and support in the AV-6 split (train / val boxes):
`person` 2533/266 · `door` 604/288 · `stairs_up` 534/46 · `dustbin` 730/30 ·
`chair` 277/28 · `table` 75/17.

`table` is the sparsest trained class (17 val boxes); its AP must be reported
beside its support, never alone. Eight further class names exist in the
annotation vocabulary and are excluded from the trained schema — see §5.3.

`signboard` was retired from the schema on 2026-08-29 after measurement: 50
boxes across 42 of 986 frames trained to AP50 0.000. An EasyOCR sweep was
evaluated as a source of automatic proposals and rejected — 11 of 75 sampled
frames carry legible text, 8 of them from a single mall video, and a text
region is not a sign extent, so such boxes would supervise text detection that
EasyOCR already performs. Retiring it costs no runtime capability: sign
*reading* runs OCR over the whole frame and never consults a detected class.
This is the same argument §5.3 makes for the other retired names, applied to
our own measurement rather than to an inherited assumption.

## 7. Evaluation protocol

* **Text detection:** ICDAR-2015 protocol (rotated IoU ≥ 0.5) — Precision,
  Recall, H-mean; compare rows: EAST 80.7, CRAFT 86.9, MOST 88.2 (published),
  **Ours [TO FILL]**.
* **Recognition:** CRR/WRR on ICDAR15 crops, pre vs post fine-tune.
* **Objects/symbols:** mAP@50, per-class AP (stairs highlighted).
* **End-to-end latency:** ms/frame on (a) laptop CPU, (b) T4, (c) HF free tier.
  Measured 2026-08-29 on the AV-6 val split, laptop CPU (i5-1145G7), 175 timed
  frames per run (`runs/eval/compare.md`):

  | stage (p50 ms) | COCO baseline | AV-6 preliminary |
  |---|---|---|
  | detect | 321 | **102** |
  | OCR | 3,896 | **3,203** |
  | total | 4,262 | **3,330** |
  | cold start (s) | 7.5 | **3.5** |

  Frame-level hazard alert: COCO P 0.875 / R 0.236 / F1 0.372 (FP 5) versus
  AV-6 preliminary P 0.987 / R 0.527 / F1 0.687 (FP 1). This is the one
  vocabulary-independent comparison; the T4 and HF-tier rows are **[PENDING]**.
* **Step distance:** MAE, RMSE, %within±1step; Bland-Altman plot.
* **Ablations:** rotation-aug off · deskew off · OBB→HBB · priority off.
* **User study (n ≥ 10, ethics approval + informed consent):** 3 tasks
  (find washroom / avoid obstacle course / describe room), measure completion
  time, collisions, SUS, NASA-TLX; baseline = same system with priority
  scheduler disabled and area-only proximity.

## 8. Results tables (fill after training)

**T1 Detection (ICDAR-2015):** P / R / H-mean / FPS — Ours-scratch, Ours-finetune, EAST, CRAFT, MOST.
**T2 Recognition:** CRR / WRR — EasyOCR-stock vs Ours-finetuned.
**T3 Obstacles:** per-class AP@50 (12 classes).
**T4 Step error:** per-distance MAE (1–8 m).
**T5 User study:** time / collisions / SUS, baseline vs full (report p-values, Wilcoxon).

## 9. Reproducibility / release checklist

☐ GitHub repo tag `v1.0-paper` ☐ trained obstacle weights (AV-6) ☐ trained
OBB text weights ☐ AV-6 split + the 219 adjudicated review-queue verdicts ☐
training configs + seeds ☐ eval scripts and run JSONs ☐ demo video.

Removed from this list: a symbol dataset and a user-study protocol, neither of
which exists. Ship the list you can satisfy; a reviewer who follows a dead
release link discounts everything else on the page.

## 10. Key references (verify/complete before submission)

Zhou et al., "EAST," CVPR 2017 · He et al., "MOST," CVPR 2021 · Liao et al.,
"Real-time Scene Text Detection with Differentiable Binarization," AAAI 2020 ·
Baek et al., "CRAFT," CVPR 2019 · Shi et al., "CRNN," IEEE TPAMI 2017 ·
Gupta et al., "SynthText," CVPR 2016 · Karatzas et al., "ICDAR 2015 Robust
Reading," ICDAR 2015 · Yao et al., "MSRA-TD500," CVPR 2012 · Lin et al.,
"Microsoft COCO," ECCV 2014 · Jocher et al., "Ultralytics YOLOv8," 2023 ·
Brooke, "SUS," 1996.
