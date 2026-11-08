# Product vision

**A personal spatial assistant for blind and visually impaired people, delivered
as a web application and installable PWA.**

The user opens it in their phone's browser, points the camera where they are
going, and asks questions in their own words. It answers from what it can
actually see, where it actually is, and what it has been told to remember.

## Web-only, decided

The end-user application is a browser/PWA application and nothing else. There
is no Android application, no Kotlin, no APK, no Play Store listing. Any earlier
requirement for native Android is superseded; this is recorded as
[R-007](../../docs/ARCHITECTURE_RULINGS.md).

Python, GPUs and server infrastructure remain in use for training, evaluation
and optional server-side reasoning. That is engineering infrastructure. It is
not the product.

## The mapping the product is built around

| faculty | implementation |
|---|---|
| eyes | phone camera via `getUserMedia` |
| visual cortex | multi-provider perception — objects, text, depth, motion |
| spatial understanding | world model of tracked entities |
| safety reflex | risk engine, which pre-empts everything else |
| reasoning | agentic brain that queries structured state, never imagines it |
| personal experience | browser-local memory of places, labels and corrections |
| macro navigation | GPS and maps |
| micro navigation | camera and spatial perception |
| voice | speech synthesis, recognition, haptics where the platform allows |
| long-term learning | dataset and training pipeline, offline |

## What the user must eventually be able to ask

"What's in front of me?" · "Is anything blocking me?" · "Where is the door?" ·
"Where are the stairs?" · "What's on my left?" · "Read what's in front of me." ·
"Read this door sign." · "Find EXIT." · "Where is the elevator?" ·
"Guide me to the entrance." · "What is approaching me?" · "Describe this room." ·
"Remember this as my office." · "Where should I go?"

Every answer is grounded in perception, navigation state, or stored memory. The
assistant does not describe the physical world from a language model's
expectations about rooms.

## What the product is not allowed to do

- **Invent the environment.** An answer with no perceptual evidence behind it is
  a wrong answer, however fluent. "I can't tell" is a correct answer.
- **Narrate every frame.** It speaks when information is new, when risk changes,
  when the user asks, or when guidance state changes. This was fixed and is now
  the baseline; see [REALTIME_FIXES.md](../../docs/REALTIME_FIXES.md).
- **Say it is safe to cross a road.** Object detection cannot support that
  claim and the product will not make it.
- **Claim a direction for a staircase it cannot determine.** The current
  detector cannot, and says so out loud (R-005).

## The honest state of it

The perception layer — the eyes — is the weakest part by a wide margin. Fused
recall is **0.297** on held-out validation images: under a third of the objects
that are annotated in frame. The architecture around it is sound and tested; the
detector is not good enough for a product people would rely on to walk safely,
and improving it is the standing first priority (§40). A quiet, well-behaved
application built on a detector that misses two thirds of obstacles is not a
product. It is a demonstration of one.
