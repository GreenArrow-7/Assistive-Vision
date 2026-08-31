"""Re-plot the PR curve and confusion matrix from the harness's own data at
one-column IEEE size, so labels stay >=7pt in print. Ultralytics' stock plots
are sized for 9-12in figures; scaled into a 3.25in column their text drops to
~3.5pt. Same numbers, readable typography — nothing is recomputed by hand.

  python docs/figures/make_curve_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

OUT = Path("docs/figures")
plt.rcParams.update({"font.family": "serif", "font.size": 7})

# plots=True is required: the confusion matrix is only accumulated on that path
m = YOLO("models/av_obstacle_candidate.pt")
r = m.val(data="datasets/av7_split/data.yaml", imgsz=640, plots=True,
          project="runs/eval", name="av7_curvedata", exist_ok=True, verbose=False)

names = [r.names[i] for i in sorted(r.names)]

# ---- PR curve -------------------------------------------------------------
# curves_results: [px, py, xlabel, ylabel] per curve; index 0 is precision-recall
px, py, xlab, ylab = r.box.curves_results[0]
py = np.asarray(py)                      # shape (nc, n_points)
ap50 = {names[int(c)]: r.box.ap50[i] for i, c in enumerate(r.box.ap_class_index)}

fig, ax = plt.subplots(figsize=(3.25, 2.35))
for i, cls in enumerate(names):
    if cls not in ap50:
        continue
    ax.plot(px, py[i], lw=0.9, label=f"{cls} {ap50[cls]:.3f}")
ax.plot(px, py.mean(0), lw=2.0, color="black",
        label=f"all classes {r.box.map50:.3f}")
ax.set_xlabel("Recall", fontsize=7.5)
ax.set_ylabel("Precision", fontsize=7.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.tick_params(labelsize=7)
ax.grid(alpha=0.25, lw=0.4)
ax.legend(fontsize=6.2, frameon=False, loc="upper right",
          handlelength=1.4, borderaxespad=0.2, labelspacing=0.25)
fig.savefig(OUT / "fig5_pr_curve.png", dpi=400, bbox_inches="tight")
plt.close(fig)

# ---- confusion matrix (normalized by true column) --------------------------
cm = r.confusion_matrix.matrix.astype(float)
labels = names + ["background"]
with np.errstate(divide="ignore", invalid="ignore"):
    cmn = cm / cm.sum(0, keepdims=True)
cmn = np.nan_to_num(cmn)

fig, ax = plt.subplots(figsize=(3.25, 2.75))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
ax.set_yticklabels(labels, fontsize=6.5)
ax.set_xlabel("True", fontsize=7.5); ax.set_ylabel("Predicted", fontsize=7.5)
for i in range(cmn.shape[0]):
    for j in range(cmn.shape[1]):
        v = cmn[i, j]
        if v >= 0.01:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if v > 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.ax.tick_params(labelsize=6)
fig.savefig(OUT / "fig6_confusion.png", dpi=400, bbox_inches="tight")
plt.close(fig)
print("re-plotted fig5, fig6 from harness data")
