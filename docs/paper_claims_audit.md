# Paper claims audit — what the draft promises vs what exists

Run before any submission. Every row marked **UNBACKED** is a claim in
`docs/ieee_paper_package.md` that no artefact in this repository supports. A
paper is not a plan: a reviewer reads the past tense as work performed, and
reporting unperformed work as performed is fabrication regardless of intent.

Repo state at audit (2026-08-29): 146 passing tests; a preliminary detector
(`models/av_obstacle_candidate.pt`, YOLOv8n/640/12ep) was trained and evaluated,
then superseded: it speaks the retired seven-class schema, so `detect_schema`
now refuses it outright — the guard working as designed. Its measurements stand
as a record of that schema (mAP@50 0.293; hazard-frame recall 0.236 -> 0.527
over the COCO baseline; `person` recall regressed, which is why it was never
deployed). The AV-6 YOLOv8s run is pending.

---

## 1. Claims with no supporting work

| # | Draft claims | Reality | Verdict |
|---|---|---|---|
| A1 | "user study with **[N]** blindfolded/low-vision participants shows task-completion time reduced by [TO FILL]% (SUS [TO FILL])" | No study designed, no ethics approval, no participants, no protocol | **UNBACKED — delete or run it** |
| A2 | CRNN recognizer "fine-tuned on 90k SynthText word crops + ICDAR15 train crops; 10 epochs, AdamW 1e-4" | Recognition is stock EasyOCR. No recognizer training code exists | **UNBACKED — delete** |
| A3 | Detector "fine-tuned on SynthText and ICDAR-2015/**MSRA-TD500**" | No MSRA-TD500 converter, no download, not in any script | **UNBACKED — drop MSRA** |
| A4 | "released **custom symbol dataset**" / *AV-Symbol-2k*, 10 classes, ~200 images/class | The five `sign_*` classes hold **zero** boxes, and `signboard` was retired at 50 boxes / AP 0.000. No symbol dataset exists | **UNBACKED — delete C5** |
| A5 | *AV-Obstacle-3k*, "**12 classes**: … vehicle, bicycle, bench, plant …" | 7 trained classes; vehicle/bicycle/bench/plant were never in the schema | **WRONG — rewrite** |
| A6 | "~3,000 images: 1,500 self-captured + 1,500 filtered Open Images" | 1,202 walkthrough frames (8 videos) + 1,451 Open Images = 2,653; split as built holds 2,076 frames | **WRONG — use real figures** |
| A7 | "**Code, trained weights** … are released" | Only a preliminary candidate exists, not deployed | **PREMATURE — gate on the final training run** |

## 2. Claims that are true and defensible

| Claim | Evidence |
|---|---|
| Perspective rectification before OCR | `server/text_pipeline.py::_deskew`, tested |
| Step-metric distance from two fused monocular estimators | `server/spatial.py::estimate_steps`, 8 tests incl. portrait regression |
| Priority scheduler ordering hazards → query → symbols → summary | `server/priority.py`, 9 tests |
| Full-lifecycle voice interface with self-echo suppression | `web/index.html`, mic paused during TTS |
| Leakage-safe by-video split | `scripts/prepare_split.py`, disjointness re-proved per run |
| Reproducible dataset pipeline | build → remap → review → merge → reindex → split, 143 tests |

## 3. The contribution the draft undersells

`server/detector.py::detect_schema` identifies a model by its **class names**,
never its filename, and refuses an unrecognised vocabulary at startup. The bug
it was written to kill: a COCO-class model placed at the custom-weights path was
assumed to be the fine-tune, which switched hazard lookup to `AV_HAZARDS` — a set
containing no vehicles — so cars, buses and trucks silently stopped being
hazards, with no error raised.

`scripts/prepare_split.py` extends the same idea to the dataset side: it refuses
a split that declares a class with no training boxes, because such a model passes
the name check while being unable to emit what it claims.

This is a *safety-by-construction* argument for assistive systems — silent
capability loss is the failure mode that matters when the user cannot see that
the system stopped warning them. It is implemented, tested, and currently absent
from the contributions list.

## 4. Required rewrites

**Abstract.** Remove the user study, the CRNN training, and MSRA-TD500. State
the trained-schema size honestly. Keep `[PENDING]` markers for numbers the
harness will produce; never fill one from literature.

**Contributions.** C1 holds only once the OBB model is trained — until then the
text detector is stock CRAFT and must be described as such. C2, C3, C4 hold now.
C5 becomes the dataset *pipeline*, not a symbol dataset. Add C6: schema
verification by class names.

**Datasets.** 1,202 frames from 8 walkthrough videos (labels tracked; 215
frames from one phone video currently lack images), 1,451 Open Images V7
images (2,722 boxes). Split as built on 2026-08-29: 1,726 train / 344 val
frames, 4,753 / 675 boxes. Re-run `scripts/prepare_split.py` and refresh
these figures after annotation; never quote the old 2,694/374 split.

**Scope.** State that the system is indoor-only and that vehicles are handled by
the COCO fallback, not the custom model. A stated boundary is a methodology
strength; an unstated one a reviewer discovers is a credibility loss.

## 5. Do not fill a `[PENDING]` from a paper

Six `[TO FILL]` markers remain. `scripts/evaluate.py` produces every one of them
from our own data, and refuses to score a COCO model against COCO-generated
pre-labels precisely so a flattering number cannot be manufactured by accident.
Numbers arrive from a run, or the claim is cut.
