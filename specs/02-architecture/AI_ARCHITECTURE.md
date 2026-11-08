# AI architecture

Two things live here: the perception stack that produces evidence, and the
conversational brain that is only ever allowed to answer from it.

## Perception is layered, and not all of it is object detection

§17 of the brief is explicit that the project must not collapse into "run YOLO".
Each layer below needs a different technology, and forcing them all into one
detector is how a class ends up declared but unlearnable.

| layer | right technology | state |
|---|---|---|
| general objects — vehicles, people, street furniture | detection | **exists**, COCO yolov8n |
| assistive objects — doors, stairs, bins, chairs | detection, fine-tuned | **exists**, AV-7 |
| text | OCR — detection then recognition | **exists**, EasyOCR + oriented-box detector |
| tracking | IoU/identity association over time | **exists**, `realtime.js` |
| depth / distance | monocular depth estimation | **not built** — currently inferred from box geometry and a calibrated vertical FOV, which is an estimate, not a measurement |
| motion / approach | derived from tracking | **exists**, drives the approach announcements |
| walkable surface, curbs, ramps, crosswalks | **segmentation**, not detection | **not built** — a kerb is a region, not an object, and a bounding box around one says nothing useful |
| scene understanding | multimodal model, on request | **not built** |

The gap analysis against real product scenarios is in
[../../docs/DATASET_STATUS.md](../../docs/DATASET_STATUS.md).

## Multi-provider detection

Settled as R-003. Several detectors run, each with its own vocabulary, and their
output is normalised into one canonical vocabulary and fused by overlap. Which
provider is believed for a class is a policy table, derived from what each model
was trained on and checked against measurement — COCO is the better person
detector despite `person` being the fine-tune's largest class, so `PERSON` fuses
both rather than trusting the specialist.

Adding a provider is a configuration edit. Removing one does not change any
other layer.

## World model

The browser holds short-term state about each tracked entity. The brain reasons
over this, never over raw frames:

```
trackId · canonicalClass · provider · confidence · boundingBox
direction · distanceBand · estimatedDistance · velocity
approaching · inPath · risk
firstSeen · lastSeen · lastAnnounced
```

`estimatedDistance` is named an estimate on purpose. It comes from box geometry
and an assumed object height, and it is wrong whenever that assumption is wrong.
Until a depth layer exists, `distanceBand` is the field downstream code should
prefer.

## Risk engine

Runs ahead of everything conversational and can interrupt it. Its output is not
advice. It will not say a road is safe to cross — detection cannot support that
claim — and it will not claim a staircase's direction while the detector cannot
determine one (R-005).

## Grounded brain

The conversational layer answers by calling tools against structured state. It
is never given the frame and asked what is in it.

```
queryWorldModel(region)      findObject(name)        readVisibleText()
describeScene()              getCurrentHazards()     getNavigationState()
getRememberedPlace(name)     startScan()             stopScan()
```

"What's on my left?" becomes `queryWorldModel("left")`, then a sentence built
from what came back. If it came back empty, the answer is that nothing was
detected on the left — which is different from "nothing is there", and should be
worded as the former.

**The hard constraint:** a language model in this product may phrase evidence and
may decline. It may not supply evidence. A fluent description of a room that was
never perceived is the worst failure this system can produce, because it is the
one the user cannot detect.

## Learning, in two separate loops

**Fast, personal, in the browser:** user labels, remembered places, corrections,
preferences. Local to the device, visible and deletable by the user.

**Slow, central, offline:** dataset → training → candidate → validation → locked
test → browser benchmark → safety review → promotion → rollback. Described in
[LEARNING_ARCHITECTURE.md](LEARNING_ARCHITECTURE.md).

These do not meet. Nothing a user teaches the app retrains the safety detector,
and no training run swaps a detector inside a live session.
