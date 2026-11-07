# Objectives and where they are implemented

The four stated project objectives, mapped to the code that carries them out. The deliverable is the
**web client** (`web/`) served by the FastAPI server (`server/`); the Android client lives in
`../Nexora` and is covered at the end. Checked on 2026-11-01 against the working tree of both folders.
Every claim cites a file and a line. Nothing was measured for this document: where a claim is about
behaviour, it points at the test or the README section that backs it.

| # | Objective | Status | Carried out by |
|---|---|---|---|
| 1 | Text recognition using OCR | Met | `server/text_pipeline.py`, text lane in `web/app.js:235` |
| 2 | Voice interaction and keyword search | Met | `web/workflow.js`, `web/speech.js`, `server/symbols.py` |
| 3 | Audio guidance and environment summary | Met, with one limit | `web/speech.js`, `web/realtime.js`, `web/app.js:265` |
| 4 | Oriented text from natural scenes | Met; own OBB detector trained and adopted | `server/text_pipeline.py:183` (Path A: OBB quads, deskew, recognize), `models/text_obb.pt` |

## 1. Text recognition using OCR

- `server/text_pipeline.py:172` `detect_text()` — EasyOCR reading, one item per region:
  `{label, box, conf, quad, angle, kind: "text"}`.
- `server/text_pipeline.py:101` `useful_text()` — drops OCR garbage: 2 to 180 characters, at least 60 %
  alphanumeric, so room numbers and short signs survive and noise does not.
- `server/config.py:12` `TEXT_CONF = 0.40` is the reading threshold; the client speaks text unasked
  only above `0.6` (`web/app.js:247`).
- The browser runs **two independent lanes**, so reading text never delays obstacle warnings:
  `web/app.js:234` (`objectLane`, `frame_interval`) and `web/app.js:235` (`textLane`,
  `ocr_interval_ms`), both built on `LatestLane` (`web/realtime.js:52`), which keeps exactly one
  request in flight per lane and never queues a backlog (`web/realtime.js:60` `_tick`).
- Each lane draws on **its own canvas** (`web/app.js:147`), because two simultaneous `toBlob` calls on
  one canvas would redraw the same pixels. Frames are scaled to 960 px and encoded at JPEG q.85
  (`web/app.js:169`).
- Intervals come from the server, so client and server stay in step: `GET /api/config`
  (`web/app.js:68`).

Backed by: `web/check.cjs` and the Node regression tests over `workflow.js`, `speech.js` and
`realtime.js`, plus the "Misread sign text 1 → 0" row in `../Nexora/README.md` (measured on the shared
announcement rules).

## 2. Voice interaction and keyword search

- `web/workflow.js:12` `command()` — the whole command parser. Curly apostrophes, trailing punctuation
  and polite fillers ("hey Nexora, please …") are stripped at line 14 before matching; fixed phrases
  are matched without punctuation while a target keeps its own, so "room 2.5" and "MG Road, Bangalore"
  survive (line 17).
- Search intents at `web/workflow.js:29` and `web/workflow.js:32`: "find", "locate", "search for",
  "where is", "show me" and friends, with trailing qualifiers ("… nearby", "… located") removed
  because they are not part of the sign being searched for. `web/workflow.js:34` refuses "where is
  **it**" and "where are **we**" — a question, not a sign.
- `web/workflow.js:46` — a bare word is accepted as a target only right after the app asked for one,
  and `web/workflow.js:44` makes sure a stray "yes" or "no" is never taken as a search target.
- Recognition is continuous: `web/speech.js:15` configures `continuous` with no interim results, and
  `web/speech.js:41` handles `not-allowed`, `audio-capture` and `network` errors — the last announced
  once per outage, then retried silently every 3 s (`web/speech.js:46`).
- The spoken keyword is posted with every text frame (`web/app.js:191`,
  `fd.append('keyword', workflow.query)`) and matched **on the server by whole word**, not substring:
  `server/symbols.py:18` `mentions()` and `server/symbols.py:84` `match_keyword()`. The comments there
  record the bugs this fixed — "find washroom" matching a MENU board through "men", "exit" matching
  "EXITED" — and synonyms resolve in both directions while "ladies" is never widened to a GENTS sign.
- A match is tracked per keyword, so an answer in flight cannot be credited to a newer target:
  `t.searchSpoken = query` in `web/app.js:254`, read back at `web/app.js:401`.

## 3. Audio guidance and environment summarization

- `web/speech.js:3` `SpeechManager` — one recognizer, one utterance, one restart timer:
  - **Priority queue.** `enqueue` at `web/speech.js:124` sorts by priority and keeps at most three
    items; priorities are assigned in `web/app.js:250` as critical 0, hazard or searched keyword 1,
    sign text 2, plain object 3. A response never cuts a critical warning (`web/speech.js:117`).
  - **The microphone is paused while speaking** (`_stopRecognition` from `_drain`,
    `web/speech.js:146`) and stays off for a 650 ms cooldown afterwards (`web/speech.js:172`), which is
    what stops the app hearing its own voice. Two further guards: events from before the current
    listening session are dropped by timestamp (`web/speech.js:25`) and the same sentence within 3 s is
    ignored (`web/speech.js:30`).
  - **A queued sentence must still be true when it starts**: `prune()` at `web/speech.js:137` drops
    expired items and anything whose `valid()` no longer holds.
- `web/realtime.js:10` `DetectionMemory` — **two sightings confirm a thing before it is said**
  (`persistence: 2`, line 11; `track.stable` at line 35, with a critical hazard exempt), tracks live
  6.5 s (`ttl`), and a thing is re-announced only when it moved by 18 % of the frame, changed by two
  steps or 35 %, or has become critical (`web/realtime.js:31`).
- Wording: `web/realtime.js:72` `describe()` — name, "approximately N steps" (singular at one step,
  "far away" past 30), direction. The distances come from `server/spatial.py:23` `estimate_steps()`
  with the browser's own tilt (`web/app.js:459`, `deviceorientation`) and `server/spatial.py:89`
  `direction()`.
- Signs versus incidental text: `web/realtime.js:79` `textIsImportant()` — the same word list and the
  same "short, mostly upper-case" rule as the Android client's
  `../Nexora/app/src/main/java/com/nexora/app/ui/Phrases.kt:151`.
- **Summaries**: `web/app.js:283` `environmentScan()` answers "what is around me" from both lanes and
  `web/app.js:265` `finishSummary()` speaks it, taking only tracks seen within the track lifetime
  (line 270). Voice phrasings for it are at `web/workflow.js:26` and `web/workflow.js:28`.
- **Not only speech**: a critical hazard also flashes and vibrates (`web/app.js:261` `flashHazard`,
  `navigator.vibrate` at line 262), and while muted, warnings and search results still reach a screen
  reader through an ARIA live region (`web/app.js:33` `show(text, live)` writing `#srLive`, decided at
  `web/app.js:255`).
- Repeated conditions are rate-limited rather than repeated: `notice()` at `web/app.js:38` says a thing
  once per 30 s by key.

One limit to state plainly: this is guidance, not routing. Indoors the client gives a direction and an
approximate step count towards a thing that is in view; it does not compute a path. Outdoors the route
is handed to Google Maps after an explicit yes (`web/app.js:383` `openMaps`, URL built in
`web/realtime.js:85` `mapsUrl`), with distance and time from `POST /api/navigation`
(`web/app.js:358`).

## 4. Oriented text from natural scenes

What is in place:

- `server/text_pipeline.py:172` — EasyOCR's CRAFT detector returns a quadrilateral per region, so
  tilted text on signboards and labels is found as a quad, not as an upright box.
- `server/text_pipeline.py:90` `_deskew()` — each quad is perspective-warped to horizontal before the
  characters are read, which is what makes an angled sign readable.
- `server/text_pipeline.py:185` — a bounded retry re-reads an uncertain region at 90°, 180° and 270°
  and keeps the better score. At most two per frame, so the cost stays fixed.
- `server/text_pipeline.py:120` `merge_text_lines()` — words on the same oriented baseline (within 12°)
  are joined in reading order, so a phrase search works on a tilted multi-word sign.
- `angle` and `quad` reach the client: `server/main.py:267` `_pub()` passes every field
  through, and the overlay strokes the quad itself when one is present (`web/app.js:502`),
  falling back to a rectangle only for items without orientation.
- The demo set includes the case on purpose: `demo/scenes.json` has `college_front`, described as "the
  college building with its tilted name board", and `exit_tilted`.

A detector of our own now exists. `scripts/train_cocotext_colab.py` converts COCO-Text v2
polygons to YOLO-OBB labels (every polygon through `cv2.minAreaRect`, illegible instances dropped)
and trains `yolov8n-obb` on 13,880 train / 3,261 val images with `degrees=30`. Measured on the
COCO-Text val split, 60 epochs, imgsz 960:

| precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.738 | 0.712 | 0.772 | 0.579 |

The labels are genuinely oriented, not axis-aligned boxes in disguise: all 80,844 are 9-field
polygons, and 24.8 % deviate from axis-aligned by more than 10°, 6.3 % by more than 30°.

`_deskew()` had a real defect until it was fixed here. YOLO-OBB's `xyxyxyxy` starts at an arbitrary
corner of the rotated rect, where CRAFT returns corners in reading order, so Path A was handing
EasyOCR crops that were mirrored or upside down. `_order_quad()` at `server/text_pipeline.py:64`
fixes the corner order for both paths. No detection metric could have caught this — the boxes were
right and the reading was empty.

Path A is adopted. The 60-epoch weights are installed at `models/text_obb.pt` and the live server
runs on them. A second defect surfaced only once they were: `detect_text` passed the raw
`xyxyxyxy` to `_quad_angle` while `_deskew` ordered the corners itself, so every reported angle was
90 degrees out -- the tilted EXIT sign came back at -78.7 deg where CRAFT said 12.6 deg. The crop
was correct and the text read, so no test and no detection metric could see it. Both now order the
quad once, at `server/text_pipeline.py:188`.

End-to-end comparison on the seven demo signs, both paths in-process, same frames:

| image | Path A | CRAFT |
|---|---|---|
| exit_tilted | `EXIT` @ 11.3 deg | `EXIT` @ 12.6 deg |
| college_front | `atme` @ 19.9 deg | `atme` @ 21.4 deg |
| pharmacy_room | `PHARMACY`, `ROOM 204` | `PHARMACY`, `ROOM 204` |
| washroom_menu | `GENTS`, `MENU`, `WASHROOM` | same |
| stairs_down_simulated | all four lines | all four lines |
| cabin_signs | `PRINCIPAL`, `Mhold` | `CABIN` |
| campus_steps | none | none |

Path A matches CRAFT on five of seven, reads a tilted sign at the right angle, and runs in about
320 ms against CRAFT's 1,900 ms. `cabin_signs` is a wash rather than a regression: the sign reads
"PRINCIPAL CABIN" and each path gets half of it.

What is still not in place:

- **The comparison is on street-scene training data and seven demo photos, not corridors.**
  COCO-Text is signage in the wild; the deployment is indoor signs photographed from a tilted
  phone. A dozen real photos would turn this into an answer about this product rather than about
  this dataset.
- **No hybrid.** When Path A misses a region CRAFT would catch, as on `cabin_signs`, nothing falls
  back. A Path A/Path B merge in `detect_text` would cover it and has not been written.

So the accurate sentence for the report is "an oriented text detector was trained on COCO-Text v2,
reaches mAP50 0.772, and is the shipped reading path; it matches the previous CRAFT pipeline on the
demo signs at roughly six times the speed and reports sign orientation", not "it is measured better
than CRAFT on indoor signage".

## Running it

```bat
run_demo.cmd          recorded scenes, no camera, no models, no internet
run.cmd               live camera and models
```

Both serve the client at **http://localhost:8000** (`server/main.py:211`). For a phone, camera and
microphone need a secure origin: either `..\Nexora\scripts\server-online.cmd` (Cloudflare quick tunnel,
trusted HTTPS, works from any network) or the LAN certificate described under "HTTPS smartphone
testing" in this repository's README. Verified on 2026-11-01: the page answered 200 and `/api/health`
reported `ready:true, demo:true` on `localhost:8000`, on the LAN at `https://192.168.0.104:8443`, and
through the tunnel.

## The Android client

`../Nexora/app/` is a second front end over the same server and the same four objectives. It is kept
because it carries evidence the browser client does not: `ReplayTest` replays three recorded walks
through the real announcement logic and the real speech queue in virtual time, with the accepted
transcripts in `../Nexora/docs/replay/` as fixtures, and the measured before/after table in
`../Nexora/README.md` comes from it. Last run on 2026-11-01 with JDK 21.0.13 and SDK platform 36:
BUILD SUCCESSFUL in 3 min 39 s, 132 tests, 0 failures, 0 skipped.

The two clients are close to equivalent. Both have the priority speech queue with the microphone paused
while speaking, two-sighting confirmation, the same sign-word list, the environment summary, the
keyword search and the Maps handoff. What only the Android client has: the Uber fare lookup and ride
handoff (`web/*.js` calls `health`, `config`, `detect`, `navigation` and `demo/scenes`, never
`/api/uber`), and the replay evidence above.
