"""Generate the paper's drawn figures (1-3) and the training-curve figure (4)
from results.csv. Figures 5-6 are ultralytics outputs copied verbatim.
Run from the repo root:  python docs/figures/make_figures.py
"""
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent
plt.rcParams.update({"font.family": "serif", "font.size": 9})


def box(ax, x, y, w, h, text, fc="#f2f2f2", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="black", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, text="", fs=8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, lw=0.9, color="black"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.012, text, ha="center",
                va="bottom", fontsize=fs, style="italic")


# ---------------------------------------------------------------- Fig. 1
fig, ax = plt.subplots(figsize=(7.0, 3.1))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

box(ax, 0.01, 0.10, 0.30, 0.80, "", fc="white")
ax.text(0.16, 0.845, "Smartphone web client", ha="center", fontsize=9.5,
        fontweight="bold")
for i, t in enumerate(["Camera capture (JPEG)", "Voice input (Web Speech)",
                       "TTS + vibration output", "Gyroscope pitch / GPS",
                       "FOV calibration (per device)"]):
    box(ax, 0.03, 0.66 - i * 0.135, 0.26, 0.105, t, fs=8)

box(ax, 0.42, 0.10, 0.57, 0.80, "", fc="white")
ax.text(0.705, 0.845, "Inference server (FastAPI, Python)", ha="center",
        fontsize=9.5, fontweight="bold")
box(ax, 0.44, 0.655, 0.25, 0.115, "YOLOv8 obstacle detector\n(schema verified by class names)", fs=7.5)
box(ax, 0.44, 0.515, 0.25, 0.115, "Oriented text: CRAFT quads\n→ deskew → CRNN (EasyOCR)", fs=7.5)
box(ax, 0.44, 0.375, 0.25, 0.115, "Symbol resolution\n(class + whole-word text)", fs=7.5)
box(ax, 0.72, 0.655, 0.25, 0.115, "Spatial: direction thirds +\nstep distance (min-fusion)", fs=7.5)
box(ax, 0.72, 0.515, 0.25, 0.115, "Priority engine\ncritical → hazard → query", fs=7.5)
box(ax, 0.72, 0.375, 0.25, 0.115, "/health capability report\n(schema, dormant hazards)", fs=7.5)
box(ax, 0.44, 0.155, 0.53, 0.14,
    "Guards: 8 MB frame cap · rate limit · warmup 503 · schema refusal",
    fc="#e8e8e8", fs=8)

arrow(ax, 0.315, 0.60, 0.415, 0.60)
ax.text(0.365, 0.635, "frame +\nkeyword", ha="center", va="bottom", fontsize=7,
        style="italic")
arrow(ax, 0.415, 0.30, 0.315, 0.30)
ax.text(0.365, 0.245, "JSON +\nspeech", ha="center", va="top", fontsize=7,
        style="italic")
fig.savefig(OUT / "fig1_architecture.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 2
fig, ax = plt.subplots(figsize=(3.5, 4.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
steps = [
    ("Detections + texts + symbols\n(one frame)", "#ffffff"),
    ("stairs_down present?\n→ “Warning — stop” preempts all", "#d9d9d9"),
    ("Hazard-class objects:\ndistant ⇒ demoted to objects", "#e8e8e8"),
    ("Proximate hazards spoken first\n(“Caution …, N steps”)", "#e8e8e8"),
    ("User-requested keyword match\n(detection route, then text route)", "#f2f2f2"),
    ("Signs / symbols", "#f2f2f2"),
    ("Scene summary", "#f2f2f2"),
    ("One composed sentence → TTS\n+ vibration on hazard", "#ffffff"),
]
n = len(steps)
for i, (t, c) in enumerate(steps):
    y = 1 - (i + 1) * (1 / (n + 0.3))
    box(ax, 0.08, y, 0.84, 0.082, t, fc=c, fs=7.6)
    if i:
        yp = 1 - i * (1 / (n + 0.3))
        arrow(ax, 0.5, yp, 0.5, y + 0.082)
fig.savefig(OUT / "fig2_priority.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 3
fig, ax = plt.subplots(figsize=(7.0, 2.9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.01, 0.72, 0.20, 0.20, "8 walkthrough videos\n(2 public, 6 self-recorded)", fs=7.5)
box(ax, 0.26, 0.72, 0.20, 0.20, "1,202 frames @ 0.5 fps\nblur/duplicate filtered", fs=7.5)
box(ax, 0.51, 0.72, 0.20, 0.20, "YOLOv8s pre-labels\n(correct, don't draw)", fs=7.5)
box(ax, 0.76, 0.72, 0.22, 0.20, "219 ambiguous boxes\nadjudicated in review tool", fs=7.5)
box(ax, 0.01, 0.40, 0.20, 0.20, "Open Images V7\n1,451 images", fs=7.5)
box(ax, 0.26, 0.40, 0.30, 0.20, "579 “Stairs” boxes re-tagged:\nmodel proposes → human verifies\n65 down / 425 up / 89 dropped", fs=7)
box(ax, 0.61, 0.40, 0.17, 0.20, "By-name reindex\nto trained schema", fs=7.5)
box(ax, 0.83, 0.40, 0.15, 0.20, "Coverage gate\n(no empty class)", fs=7.5)
box(ax, 0.26, 0.06, 0.46, 0.22,
    "Leakage-safe split — by whole video, never by frame\n"
    "1,685 train / 336 val frames; OI keeps its upstream split", fs=8)
arrow(ax, 0.21, 0.82, 0.26, 0.82); arrow(ax, 0.46, 0.82, 0.51, 0.82)
arrow(ax, 0.71, 0.82, 0.76, 0.82)
arrow(ax, 0.21, 0.50, 0.26, 0.50); arrow(ax, 0.56, 0.50, 0.61, 0.50)
arrow(ax, 0.78, 0.50, 0.83, 0.50)
arrow(ax, 0.87, 0.72, 0.695, 0.61)          # adjudicated corpus -> reindex
arrow(ax, 0.905, 0.40, 0.72, 0.28)          # coverage gate -> split
fig.savefig(OUT / "fig3_dataset.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 4
res = Path("runs/detect/runs/local/av7_baseline/results.csv")
rows = list(csv.DictReader(res.open()))
g = lambda r, k: float(next(v for c, v in r.items() if k in c))
ep = [int(float(r["epoch"])) for r in rows]
m50 = [g(r, "mAP50(B)") for r in rows]
m95 = [g(r, "mAP50-95") for r in rows]
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.plot(ep, m50, "-", color="black", lw=1.2, label="mAP@50")
ax.plot(ep, m95, "--", color="#555555", lw=1.2, label="mAP@50–95")
b = max(range(len(m50)), key=lambda i: m50[i])
ax.plot(ep[b], m50[b], "o", color="black", ms=4)
ax.annotate(f"best {m50[b]:.3f} (ep {ep[b]})", (ep[b], m50[b]),
            textcoords="offset points", xytext=(-8, 6), fontsize=7.5)
ax.set_xlabel("epoch"); ax.set_ylabel("validation mAP")
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25, lw=0.4)
fig.savefig(OUT / "fig4_training.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig. 5-6
src = Path("runs/detect/runs/eval/av7_curves")
shutil.copy(src / "BoxPR_curve.png", OUT / "fig5_pr_curve.png")
shutil.copy(src / "confusion_matrix_normalized.png", OUT / "fig6_confusion.png")
print("figures written to", OUT)
