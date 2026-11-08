# Implementation plan

Priority order is the one set in §44 of the brief. What changed in this
iteration is marked; everything else is open.

| P | work | state |
|---|---|---|
| **P0** | Preserve AV-7 and the datasets | **done** this iteration |
| **P1** | Correct dataset and evaluation methodology | **done** — terminology fixed, source-level leakage check built, `assistive_test_v1` designed but not built |
| **P2** | Audit dataset gaps | **done** — DATASET_STATUS.md |
| **P3** | Improve perception models | **open, and the priority** |
| P4 | Browser-compatible model export | **done** — ONNX, measured |
| P5 | Multi-provider web inference | open — the fusion layer exists server-side; no browser runtime |
| P6 | Benchmark on a real mobile browser | open — blocked on a deployed origin and a phone |
| P7 | Improve OCR | open |
| P8 | Tracking, depth, motion, world model | partial — tracking and motion exist; depth and a formal entity schema do not |
| P9 | Grounded conversational brain | open |
| P10 | Personal memory | open |
| P11 | Navigation | partial |
| P12 | Controlled learning and personalization | open |
| P13 | Security, accessibility, performance hardening | open |
| P14 | Field validation | open |
| P15 | Production web/PWA release | open |

## What was done in this iteration

**P0 — preservation.** `datasets/av7_split` and `datasets/av7_merged` copied out
of the Claude worktree into the working copy; nothing deleted. AV-7 and AV-6
registered in `models/registry/`, keyed by the class names and training date read
out of the checkpoint rather than by filename — the failure that cost a day
(R-002). `scripts/register_model.py` and `scripts/dataset_manifest.py` make both
repeatable.

**P1 — methodology.** `av7_split/val` is now called validation data everywhere
(R-008). The leakage check was replaced with one that holds for video frames:
grouping by source recording rather than hashing bytes. The result is zero
shared recordings, so the existing figures survive the stronger test — the fix
was to the method, not to the numbers.

**P2 — gap analysis.** Scenario by scenario, with the technology each gap needs.
The useful finding is that several "missing classes" are not detection problems
at all: walkable surface and kerbs need segmentation, glass doors need depth,
and entrances and signage are OCR the project already has and has not wired to
the world model.

**P4 — export.** Both providers export to ONNX and run **faster** under ONNX
Runtime than under PyTorch. The download, 21.6 MB gzipped for both, is the real
cost of in-browser inference.

**R-007 — Android closed.** Scope change recorded; the toolchain blocker is
closed rather than fixed, and no further time goes into it.

## What to do next, and why in this order

**1. P3, perception.** Recall 0.297 is the product's defining problem and
everything below it is building on sand. Concretely, and in order of evidence
per hour spent: carve the indoor part of `assistive_test_v1` from recordings
already on disk; film outdoor and vehicle footage, since no outdoor claim this
project makes has any measurement behind it; try a larger backbone, because
`yolov8n` at 3M parameters is the smallest model in the family and the export
numbers show headroom in the latency budget; then `table`, `stairs_down` and the
occlusion cases.

**2. A deployed origin.** Everything at P6 and beyond is blocked on it, and
§37 forbids a developer laptop in the production path. This is also the smallest
piece of infrastructure work on the list.

**3. The web platform.** A build, TypeScript, a worker boundary, a manifest and
a service worker. Build it around the code the audit marked KEEP — tracking,
announcement policy, sentence construction, spatial reasoning all port
unchanged — rather than rewriting it.

Deliberately **not** next: more UI, the conversational brain, personal memory.
Each would be built on a detector that misses two thirds of what is in front of
it, and a well-built application over a weak detector is the failure mode §40
warns about.
