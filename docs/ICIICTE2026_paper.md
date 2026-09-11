# ICIICTE-2026 submission — FINAL, double-blind (Track 3: AI, Data Science & Emerging Technologies)

**SUBMISSION VERSION.** Double-blind: no author names, affiliation,
acknowledgment, or repository URL appear anywhere in the manuscript. Every
number below is measured and exists in the repository (`runs/eval/*`,
`results.csv`, the test suite). Page budget: 6 pages including references.

---

# Hazard-First Assistive Vision: A Priority-Scheduled, Schema-Verified Perception Pipeline for Visually Impaired Navigation

**Abstract** — Independent indoor movement confronts a visually impaired person
with information that arrives silently: a descending staircase, an obstacle at
walking height, a sign that is rotated or perspective-distorted. Systems that
only read text or only detect objects address fragments of this problem, and
most speak their findings in arrival order rather than in order of danger. We
present an assistive vision system, implemented as a smartphone web client and
a Python inference server, that integrates oriented scene-text reading,
obstacle detection over a navigation-specific class schema, monocular distance
estimation expressed in walking steps, and a priority scheduler that speaks
hazards before requested objects, signs, and scene summaries. Two design
decisions distinguish the system. First, *safety by construction*: the runtime
identifies detector weights by their class names and refuses unrecognized
vocabularies, and the dataset tooling refuses a split that declares a class it
cannot populate. Second, *honest capability reporting*: the health endpoint
discloses which hazard classes the deployed weights can and cannot raise. Two
compact fine-tuned detectors were trained and fully evaluated against COCO
baselines on their own held-out splits: the seven-class model — trained after
a human-verified re-tagging protocol gave the safety-critical
descending-stairs class its first labelled data — raises frame-level hazard
recall from 0.297 to 0.554 at precision 0.976. Both fine-tunes, however,
regress on person recall at the deployed operating point, so neither is
deployed: per-class evaluation, not the improved aggregate, governs that
decision, and we report the regression as a finding. The pipeline itself is
validated end-to-end by 152 automated tests.

**Keywords** — assistive technology; visually impaired navigation; object
detection; scene text recognition; hazard prioritization; monocular distance
estimation; system safety verification.

## I. INTRODUCTION

The World Health Organization estimates that more than two billion people live
with a vision impairment, a substantial fraction of whom cannot resolve the
visual cues that sighted pedestrians use continuously: door frames, staircase
edges, overhead signage, and the position of people moving nearby [1].
Indoors, where satellite positioning is unavailable and tactile paving is
rare, the white cane covers roughly a one-metre radius at floor level and
nothing above it, and it reads no text at all.

Computer vision can, in principle, supply the missing information. Modern
object detectors localize dozens of object categories in real time [9], [17],
and scene-text systems detect and recognize signage under rotation and
perspective distortion [2]–[5]. In practice, however, assistive deployments of
these components tend to inherit two structural weaknesses. The first is
fragmentation: reading applications read, detection applications detect, and
neither orders its output by consequence, so the user hears about a menu board
and an approaching staircase with equal urgency. The second is silent
capability loss: when a detector is swapped or retrained, nothing in a typical
serving stack verifies that the new weights can still raise every warning the
system's logic assumes — a failure mode that is invisible precisely to the
user it endangers.

This paper describes an end-to-end system built around those two
observations. A smartphone browser captures frames and voice commands; a
server runs detection, oriented text reading, and spatial reasoning; a
priority engine composes a single spoken sentence in which danger always
precedes convenience. The class schema is navigation-specific — it names
staircases, doors, and dustbins that COCO-trained detectors structurally
cannot report — and the runtime treats the schema as a contract to be
verified, not a filename to be trusted.

Our contributions are: **(1)** a priority-scheduled perception-to-speech
pipeline in which hazards interrupt, a critical class (descending stairs)
preempts everything, and distant hazard-class objects are demoted so that
only proximate danger interrupts the user; **(2)** step-metric monocular
distance estimation that fuses two independent estimators — a class-height
pinhole model and a ground-plane model with device-pitch correction — by
taking the conservative minimum, with a one-tap per-device field-of-view
calibration; **(3)** a safety-by-construction mechanism spanning runtime and
tooling: name-based schema verification that refuses unrecognized weights,
split construction that refuses unlearnable classes, and a health endpoint
that reports which alerts the deployed model can actually raise; **(4)** a
reproducible, leakage-safe dataset pipeline for the navigation schema,
including a human-verified re-tagging protocol that gave the safety-critical
descending-stairs class its first labelled data, and a measured seven-class
detector trained on it — together with the per-class evaluation discipline
that keeps both fine-tunes out of deployment despite improved aggregates.
All pipeline logic is pinned by 152 automated tests.

## II. RELATED WORK

**Assistive vision systems.** Surveys of assistive technology for blind and
low-vision users trace a progression from sonar canes to camera-based scene
description [6], [7]. Deployed reading assistants demonstrate demand, but
published descriptions emphasize recognition accuracy over information
ordering: output is typically spoken in detection order, and obstacle
awareness, when present, is not integrated with text reading in a single
prioritized channel. Navigation-focused prototypes conversely emphasize
obstacle avoidance and path planning [7] but rarely read signage, although
signage is how buildings communicate routes to everyone else.

**Scene-text detection and recognition.** EAST [2] and CRAFT [3] established
efficient oriented and character-affinity text detection; DBNet [4] and MOST
[5] refined arbitrary-shape detection. Recognition commonly follows the CRNN
design [10]. These systems target benchmark protocols, not assistive
delivery: they stop at bounding quadrilaterals and transcripts. Our pipeline
consumes their output — each detected quadrilateral is
perspective-rectified before recognition — and treats reading as one
prioritized voice among several, not as the product.

**Object detection for navigation classes.** The single-stage detection
lineage from YOLO [17] to YOLOv8 [9] makes obstacle detection practical on
modest hardware, but the standard vocabularies do not serve navigation. COCO
[8] supplies person, furniture, and vehicle categories with massive
supervision, yet contains no staircase, door-as-obstacle, or dustbin class;
Open Images [11] contains a single undirected *Stairs* class. Since a
descending staircase is the highest-consequence indoor hazard and an
ascending one is merely an obstacle, directionality is not a labelling
nicety but the difference between "caution" and "stop." To the best of our
knowledge, no prior assistive work both trains a direction-aware stairs
class and verifies at runtime that the deployed weights still carry it —
the gap this paper addresses. Table I positions the work.

**TABLE I — Positioning relative to representative approaches**

| Approach | Text | Obstacles | Prioritized speech | Capability verification |
|---|---|---|---|---|
| Reading assistants (e.g., [6]) | yes | no | no | no |
| Navigation prototypes [7] | no | yes | partial | no |
| Text-detection literature [2]–[5] | yes | — | — | — |
| This work | yes (oriented) | yes (nav schema) | yes | yes |

## III. PROPOSED SYSTEM

The system is client–server (Fig. 1). The client is a browser application on
an ordinary smartphone: it captures JPEG frames (continuous scan every
≈2.6 s in live-assist mode, or single-shot), performs speech recognition for
commands, renders text-to-speech and vibration, and reads the device
gyroscope and GPS. The server, a Python/FastAPI process, exposes one
analysis endpoint: a frame plus an optional keyword returns detected
objects, hazards, texts, symbols, spatial attributes, and a single composed
speech string. Outdoor turn-by-turn navigation is delegated to a maps
application by design; the contribution of this work is indoor perception.

*Fig. 1. Overall architecture of the proposed system.*
(`docs/figures/fig1_architecture.png`)

Server-side processing has five stages: (i) decode and validate the frame;
(ii) detect obstacles over the active class schema; (iii) detect and read
scene text; (iv) resolve symbols and compute direction and step distance for
every detection; (v) compose one prioritized sentence. The server also
defends its own availability — oversize frames are refused, request rate is
limited, and a failed model load reports unhealthy rather than degrading
silently — and its health endpoint reports the active schema and which alert
paths the loaded weights can serve (§IV-D).

## IV. METHODOLOGY

### A. Obstacle detection over a verified schema

A YOLOv8 detector [9] runs over the full frame. The serving layer is
schema-agnostic: if fine-tuned weights are present, their class vocabulary
is read from the weights themselves; otherwise the stock COCO model serves
as a fallback with COCO hazard semantics. The trained schema (AV-7) names
seven classes chosen for navigation consequence: person, chair, table, door,
stairs_up, dustbin, and stairs_down. Hazard status is a semantic role
assigned in code to a class name — deliberately not a visual class, since a
generic "obstacle" category has no consistent appearance for annotators to
agree on. YOLOv8 was selected for its single-stage speed on CPU-class
hardware and mature tooling; no architectural novelty is claimed for the
detector itself.

### B. Oriented text reading and symbol resolution

Text detection uses the CRAFT detector via EasyOCR [12], which yields
rotated quadrilaterals natively; each quadrilateral is perspective-warped to
horizontal before recognition by the stock CRNN recognizer [10]. No
recognizer fine-tuning is claimed. The pipeline is detector-agnostic: a
YOLOv8-OBB text detector can replace CRAFT by placing weights at a
designated path, and conversion tooling for ICDAR-2015 [13] and SynthText
[15] is implemented and tested; training such a detector is future work.

Signs resolve through two routes: a detected class implies its symbol, and
recognized text is matched against a symbol vocabulary by whole-word
comparison. The whole-word rule exists because substring matching once
announced a restaurant *MENU* board as a washroom — "men" is a substring of
both MENU and WOMEN — and the regression is pinned by a test. Five pictogram
sign classes exist in the annotation vocabulary but hold no labelled boxes
yet, so no trained symbol recognition is claimed; text-free pictograms are
the stated next annotation target precisely because they are the case
OCR-only readers cannot serve.

### C. Spatial awareness in walking steps

Direction is derived from frame thirds (left / ahead / right). Distance uses
two independent monocular estimators grounded in standard projective
geometry [14]: (i) a pinhole model over per-class real-world height priors,
and (ii) a ground-plane model that projects the bounding-box foot point
through the camera's vertical field of view, corrected by the device pitch
from the gyroscope. The two estimates are fused by taking the **minimum**:
under-estimating distance is the safe error for a person who cannot
visually confirm the estimate. The result is spoken in walking steps — the
unit the user can act on — rather than metres. Because browser-reported
field of view is unreliable across devices, the client includes a one-tap
calibration: the user places any known object four steps ahead, and the
true vertical field of view is solved from that single measurement and
persisted per device.

### D. Safety-priority engine and verified capability

The priority engine composes one sentence per frame in a fixed order:
**critical alert → hazards → user-requested object → symbols/signs → scene
summary**. The single critical class, stairs_down, produces an interrupting
"Warning — stop and proceed carefully" utterance that preempts all other
content. Hazard-class objects that are distant are demoted to ordinary
objects so that only proximate danger interrupts; the client additionally
vibrates on hazards and suppresses its own microphone while speaking, so
the recognizer does not transcribe the system's own output.

Two verification mechanisms make this pipeline trustworthy across model
updates. First, the runtime identifies any weights placed at the
custom-model path by their **class names**, never the filename. Weights
whose vocabulary matches the navigation schema activate navigation-hazard
semantics; COCO weights retain COCO hazard semantics; anything else is
refused at startup. The bug this kills is concrete: a COCO model mistaken
for the fine-tune would silently stop flagging vehicles, with no error
raised to a user who cannot see the difference. Second, the same philosophy
is applied *before* training: the split builder refuses a dataset that
declares a class with no training boxes, because the resulting model would
pass the name check while structurally unable to emit the class it claims.
Between the two checks, the health endpoint reports the active schema,
whether the critical-alert path is currently servable, and the hazard roles
the deployed weights cannot raise — capability loss is surfaced as data,
not discovered by accident.

## V. DATASET AND EXPERIMENTAL SETUP

The dataset pipeline (Fig. 2) combines self-collected walkthrough footage
with public supervision, under two rules: labels move between vocabularies
by class *name*, never by index, and no frame-level split is ever taken.

*Fig. 2. Dataset preparation pipeline with the human-verified stairs
re-tagging protocol.* (`docs/figures/fig3_dataset.png`)

**Corpus.** The self-collected corpus comprises 1,202 frames extracted at
0.5 fps from eight indoor walkthrough videos (malls, hospitals, campuses:
two public tours, six self-recorded), blur- and duplicate-filtered, and
pre-labelled by a stock YOLOv8s so that annotation is correction rather
than drawing; 219 ambiguous pre-label boxes were individually adjudicated
in a purpose-built review tool. This is supplemented by 1,451 Open Images
V7 [11] images supplying volume for door, staircase, and dustbin.

**A human-verified critical class.** Open Images labels staircases without
direction. All 579 imported staircase boxes were re-tagged under a
two-stage protocol: a vision-model pass proposed a direction for every box,
and a human then reviewed *every box proposed as descending*, confirming
65, demoting 53, and dropping one — the audit record ships with the
dataset. The asymmetric protocol reflects asymmetric risk: a wrongly
ascending label leaves the status quo, while a wrongly descending label
would train the class that triggers the system's only interrupting alert.
The 45% correction rate on proposed descending boxes is itself evidence
that unreviewed model labels would have been unacceptable for this class.
The final corpus carries 65 verified descending-stairs boxes (60 train /
5 validation) — few, but the first supervised signal this class has had.

**Split and training.** Train/validation splitting is by whole video, never
by frame: consecutive 0.5 fps frames are near-duplicates, and a frame-level
split leaks validation into training. Open Images contributes its own
upstream train/validation assignment, which is honoured rather than
re-split; four self-recorded videos are held out entirely for validation.
Table II summarizes the seven-class (AV-7) split. An earlier six-class
(AV-6) schema stage — before the stairs re-tagging enabled stairs_down —
uses the same corpus and methodology and serves as a second measured
configuration. Both fine-tunes train YOLOv8n at 640 px on a laptop CPU
(batch 8, 40 epochs; the AV-6 run early-stopped at epoch 36, the AV-7 run's
best checkpoint is epoch 29) with moderated augmentation: mosaic 0.5
(disabled for the final two epochs), rotation ±5°, value jitter 0.4,
horizontal flip 0.5. Heavier augmentation was deliberately avoided — a
short schedule spends its epochs on distorted frames instead of the real
domain. Seven further vocabulary names (five pictogram signs, pole,
signboard) hold too few boxes to train; signboard was retired from the
schema after 50 boxes measured AP@50 of 0.000, since a
declared-but-unlearnable class costs macro-mAP while the name-based check
still certifies the weights.

**TABLE II — Dataset summary (AV-7 split; boxes per class)**

| | Frames | pers. | door | dust. | st-up | chair | table | st-down |
|---|---|---|---|---|---|---|---|---|
| Train | 1,685 | 2,533 | 604 | 730 | 399 | 277 | 75 | 60 |
| Val. | 336 | 266 | 288 | 30 | 27 | 28 | 17 | 5 |

**Environment.** Python server (FastAPI, PyTorch CPU, Ultralytics YOLOv8,
EasyOCR); browser client in plain HTML/JavaScript (Web Speech, vibration,
device-orientation, geolocation APIs). Training, evaluation, and all
latency measurements ran on an Intel Core i5-1145G7 laptop CPU with no
discrete GPU; the system also ships as a container image verified
end-to-end, and the test suite runs in continuous integration on an
independent platform.

## VI. RESULTS AND DISCUSSION

### A. Functional validation

The complete pipeline is operational end-to-end with stock weights: live
capture, detection, oriented text reading, symbol resolution, step-distance
estimation, prioritized speech, voice keyword search, and the calibration
flow all function against the deployed server, inside and outside the
container. All deterministic pipeline logic is pinned by **152 automated
tests** spanning priority ordering and the critical interrupt, hazard
demotion by proximity, step-distance geometry (including a portrait-mode
regression), deskew, symbol whole-word matching, API limits, schema
verification for matching, COCO, and unknown vocabularies, and every
dataset tool: remapping, adjudication merge, by-name reindexing,
leakage-safe splitting, and coverage gating. Several tests encode bugs that
actually occurred — the MENU/WOMEN substring confusion, review-tool
verdicts silently lost to a single-threaded server — so that none can
return unnoticed.

### B. Detection results

Both fine-tunes were evaluated by the repository's own harness on held-out
validation splits, and each is paired with a COCO baseline measured on its
**own** split: the AV-6 pair shares one split, and the AV-7 pair shares the
extended split of Table II. Table III reports both pairs; cross-split
numbers are never compared directly.

**TABLE III — Measured results, each pair on its own held-out split (laptop CPU)**

| Metric | COCO | AV-6 | COCO† | AV-7† |
|---|---|---|---|---|
| Hazard precision (frame) | 0.860 | 0.958 | 0.936 | **0.976** |
| Hazard recall (frame) | 0.252 | 0.463 | 0.297 | **0.554** |
| Hazard F1 (frame) | 0.389 | 0.624 | 0.451 | **0.707** |
| False-alarm frames | 6 | 3 | 3 | **2** |
| mAP@50 | n/a (vocab.) | 0.334 | n/a (vocab.) | 0.365 |
| mAP@50–95 | n/a (vocab.) | 0.208 | n/a (vocab.) | 0.250 |
| Detection latency p50 (ms) | 119 | 96 | 197* | 163* |
| End-to-end latency p50 (ms) | 3,576 | 3,282 | 4,253* | 4,042* |

†Extended (AV-7) split. *The AV-7-split pair was measured back-to-back
under residual system load; the two columns of each pair are internally
comparable, and the AV-6-split pair was measured on an idle CPU.

The frame-level hazard alert — a frame counts as positive when it holds a
proximate hazard-class box, exactly the condition under which the server
interrupts the user — is the one row comparable across vocabularies, and it
is the row that matters for the person walking: on the same split, the AV-7
model raises recall from 0.297 to 0.554 at higher precision, and the AV-6
pair shows the same shape. Fig. 3 shows the AV-7 training trajectory;
Fig. 4 and Fig. 5 show its precision–recall curves and normalized confusion
matrix. Table IV reports per-class results.

*Fig. 3. AV-7 training: validation mAP versus epoch (YOLOv8n, 640 px,
CPU). Best checkpoint at epoch 29.* (`docs/figures/fig4_training.png`)

**TABLE IV — AV-7 per-class results (validation; ⚠ = under 30 boxes, treat
as directional)**

| Class | AP@50 | AP@50–95 | Deployed P | Deployed R | n_val |
|---|---|---|---|---|---|
| dustbin | 0.721 | 0.543 | 0.381 | 1.000 | 30 |
| stairs_down ⚠ | 0.595 | 0.515 | 1.000 | 0.333 | 5 |
| door | 0.367 | 0.226 | 0.548 | 0.302 | 288 |
| stairs_up ⚠ | 0.359 | 0.138 | 0.625 | 0.357 | 27 |
| chair ⚠ | 0.299 | 0.209 | 0.375 | 0.250 | 28 |
| person | 0.160 | 0.092 | 0.739 | 0.108 | 266 |
| table ⚠ | 0.052 | 0.026 | 0.500 | 0.125 | 17 |

*Fig. 4. Precision–recall curves of the AV-7 detector.*
(`docs/figures/fig5_pr_curve.png`)

*Fig. 5. Normalized confusion matrix of the AV-7 detector.*
(`docs/figures/fig6_confusion.png`)

**The critical class detects.** With 60 verified training boxes,
stairs_down reaches AP@50 0.595 at precision 1.000 / recall 0.527
(capacity), and at the deployed operating point raises one of its three
validation instances with zero false positives. Five validation boxes sit
far below statistically meaningful support — we report the number as a
direction, not a result — but the direction is that the system's only
interrupting alert path is, for the first time, backed by a detector that
has seen its hazard.

**A negative result, reported rather than buried.** In both fine-tunes,
person recall at the deployed operating point regresses against the COCO
baseline (0.164 → 0.061 for AV-6; 0.108 for AV-7): the fine-tuning corpus
holds 2,533 indoor person boxes against COCO's millions. The aggregate
hazard metric improves regardless, because newly detected classes
compensate — which is exactly why reporting only the aggregate would
misrepresent a system materially worse at detecting people, the most common
moving hazard indoors. Both fine-tunes are therefore **withheld from
deployment**, and the running system retains the COCO configuration that
the completed evaluation supports. Analysis of the measured curves
indicates the deployed per-class confidence threshold (0.62, calibrated for
COCO weights) accounts for part of the regression — the AV-7 model's
capacity recall for person is 0.135 against 0.108 deployed — so per-class
threshold recalibration, together with more person supervision, is the
identified path to a deployable fine-tune.

### C. Discussion

Measured end-to-end latency (p50 ≈ 3.3 s per frame on an idle laptop CPU)
is dominated by text recognition (≈3.1 s); detection contributes under
0.1 s. The system is honestly a paced assistant, not a real-time one, and
the latency target for future optimization is the OCR stage, not the
detector. The hazard-first ordering is what makes a 3-second pipeline
usable: the most consequential information is always the first thing
spoken. The person-recall regression demonstrates why per-class gates
matter more than aggregate metrics in safety-adjacent systems — and why
this system's deployment decision is made per class, not per headline
number.

## VII. LIMITATIONS AND FUTURE WORK

The limitations are stated by measurement, not disclaimer. (i) End-to-end
latency is OCR-bound at ≈3 s per frame on CPU; faster text detection —
including the YOLOv8-OBB path whose conversion tooling already ships — and
OCR gating are the highest-value optimizations. (ii) The person class
regresses at the deployed operating point in both fine-tunes; no fine-tune
ships until per-class recalibration and additional person supervision
restore it. (iii) stairs_down has five validation boxes; its AP is
directional until more descending footage is annotated — self-recorded
top-of-staircase captures are the only realistic source, since public
datasets photograph staircases from below. (iv) Five pictogram sign classes
and two further obstacle classes hold no labels yet; annotating them
re-activates trained symbol recognition through a one-line schema change,
by design. (v) Distance is monocular with a ground-plane assumption; metric
depth models are a candidate upgrade. (vi) No user study has been conducted
and none is claimed; a protocol (task completion, collisions, usability
scoring [16] against a priority-disabled baseline) is specified as future
work. (vii) Both fine-tunes are compact CPU-trained models; larger-capacity
training on the same verified corpus is a natural extension but is not part
of this work's claims.

## VIII. CONCLUSION

We presented an assistive vision system that treats *what to say first* and
*whether the model can still say it* as first-class engineering problems.
The implemented pipeline integrates oriented text reading,
navigation-specific obstacle detection, step-metric monocular distance, and
prioritized speech, and wraps the model lifecycle in verification: schemas
are checked by name at startup, unlearnable classes are refused before
training, and the health endpoint reports precisely which alerts the
deployed weights can raise. Two measured fine-tunes roughly double
frame-level hazard recall over COCO baselines on their own splits at higher
precision; the seven-class model gives the safety-critical
descending-stairs class its first detector, trained on labels that survived
an asymmetric human-verification protocol which corrected 45% of the
machine's proposals for the class that matters most. The same evaluation
discipline that produced those gains also surfaced a person-recall
regression at the deployed operating point — the documented reason neither
fine-tune is deployed and the running system retains its COCO
configuration. In a system whose user cannot visually second-guess it,
we consider that refusal, and the per-class evidence behind it, as much a
result as the improvements.

## REFERENCES

[1] World Health Organization, *World Report on Vision*. Geneva: WHO, 2019.

[2] X. Zhou et al., "EAST: An efficient and accurate scene text detector,"
in *Proc. IEEE CVPR*, 2017, pp. 5551–5560.

[3] Y. Baek, B. Lee, D. Han, S. Yun, and H. Lee, "Character region
awareness for text detection," in *Proc. IEEE CVPR*, 2019, pp. 9365–9374.

[4] M. Liao, Z. Wan, C. Yao, K. Chen, and X. Bai, "Real-time scene text
detection with differentiable binarization," in *Proc. AAAI*, 2020,
pp. 11474–11481.

[5] M. He et al., "MOST: A multi-oriented scene text detector with
localization refinement," in *Proc. IEEE CVPR*, 2021, pp. 8813–8822.

[6] A. Bhowmick and S. M. Hazarika, "An insight into assistive technology
for the visually impaired and blind people: State-of-the-art and future
trends," *J. Multimodal User Interfaces*, vol. 11, no. 2, pp. 149–172,
2017.

[7] S. Real and A. Araujo, "Navigation systems for the blind and visually
impaired: Past work, challenges, and open problems," *Sensors*, vol. 19,
no. 15, p. 3404, 2019.

[8] T.-Y. Lin et al., "Microsoft COCO: Common objects in context," in
*Proc. ECCV*, 2014, pp. 740–755.

[9] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," 2023.
[Online]. Available: https://github.com/ultralytics/ultralytics

[10] B. Shi, X. Bai, and C. Yao, "An end-to-end trainable neural network
for image-based sequence recognition and its application to scene text
recognition," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 39, no. 11,
pp. 2298–2304, 2017.

[11] A. Kuznetsova et al., "The Open Images Dataset V4: Unified image
classification, object detection, and visual relationship detection at
scale," *Int. J. Comput. Vis.*, vol. 128, pp. 1956–1981, 2020.

[12] JaidedAI, "EasyOCR: Ready-to-use OCR," 2020. [Online]. Available:
https://github.com/JaidedAI/EasyOCR

[13] D. Karatzas et al., "ICDAR 2015 competition on robust reading," in
*Proc. ICDAR*, 2015, pp. 1156–1160.

[14] R. Hartley and A. Zisserman, *Multiple View Geometry in Computer
Vision*, 2nd ed. Cambridge, U.K.: Cambridge Univ. Press, 2004.

[15] A. Gupta, A. Vedaldi, and A. Zisserman, "Synthetic data for text
localisation in natural images," in *Proc. IEEE CVPR*, 2016,
pp. 2315–2324.

[16] J. Brooke, "SUS: A 'quick and dirty' usability scale," in *Usability
Evaluation in Industry*. London, U.K.: Taylor & Francis, 1996,
pp. 189–194.

[17] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look
once: Unified, real-time object detection," in *Proc. IEEE CVPR*, 2016,
pp. 779–788.
