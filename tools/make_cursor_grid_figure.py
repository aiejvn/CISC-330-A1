"""
make_cursor_grid_figure.py - report figure for Task 6: four selected markers (one per DRR) in the cursor view.
Each panel is a 7.6 mm zoom with the cursor crosshair at the visual reading, the algorithm's result (predicted) and
the true projection (actual) drawn on it, and a status line with the coordinates under the cursor.
Uses outputs/task6_visual_readings.csv. Writes cisc330_a1_g1/figures/task6_cursor_compare.png.
Run from the A1 folder:   python tools/make_cursor_grid_figure.py
"""
import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

rows = list(csv.DictReader(open(os.path.join(config.OUTPUT_DIR, "task6_visual_readings.csv"))))
px, H = config.PIXEL_SIZE_mm, 38
# (phase, pose, marker index in the csv): the largest visual error on the sphere (B), the best case (A),
# and on the vertebra the largest visual error (A) and the largest algorithm error, in front of the bone (B)
CASES = [("sphere", "A", 2), ("sphere", "B", 2), ("vertebra", "A", 0), ("vertebra", "B", 1)]
fig, axs = plt.subplots(2, 2, figsize=(9.0, 9.6))
for ax, (phase, pose, m) in zip(axs.ravel(), CASES):
    imgs = np.load(os.path.join(config.OUTPUT_DIR, f"drr_{phase}_{pose}.npy"))
    n = imgs.shape[0]
    r = [x for x in rows if x["phase"] == phase and x["pose"] == pose][m]
    vis = np.array([float(r["visual_i1"]), float(r["visual_i2"])])
    alg = np.array([float(r["algo_i1"]), float(r["algo_i2"])])
    tru = np.array([float(r["true_i1"]), float(r["true_i2"])])
    ax.imshow(imgs, cmap="gray_r", origin="lower", extent=[0, n, 0, n], interpolation="nearest")
    ax.set_xlim(vis[0] - H, vis[0] + H); ax.set_ylim(vis[1] - H, vis[1] + H)
    ax.axvline(vis[0], color="tab:blue", lw=0.7); ax.axhline(vis[1], color="tab:blue", lw=0.7)
    ax.plot(*vis, "x", color="tab:blue", ms=11, mew=1.8)
    ax.plot(*alg, "+", color="red", ms=16, mew=1.4)
    ax.plot(*tru, "o", mfc="none", mec="limegreen", ms=20, mew=1.4)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{phase}, detector {pose}: visual {float(r['visual_vs_true_px']):.2f} px, "
                 f"predicted {float(r['algo_vs_true_px']):.2f} px", fontsize=8)
    ax.text(0.5, -0.13, f"cursor ({vis[0]:.0f}, {vis[1]:.0f}) px | u={vis[0] * px - 100:.2f} v={vis[1] * px - 100:.2f} mm\n"
            f"predicted ({alg[0]:.2f}, {alg[1]:.2f})  actual ({tru[0]:.2f}, {tru[1]:.2f})",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5, family="monospace",
            bbox=dict(boxstyle="square", fc="#eeeeee", ec="grey"))
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker="x", color="tab:blue", ls="", ms=9, mew=1.8, label="visual (cursor)"),
                    Line2D([], [], marker="+", color="red", ls="", ms=11, mew=1.4, label="predicted: algorithm"),
                    Line2D([], [], marker="o", mfc="none", mec="limegreen", ls="", ms=11, mew=1.4, label="actual: true projection")],
           loc="upper center", ncol=3, fontsize=9, frameon=False)
fig.tight_layout(h_pad=3.0, w_pad=2.0, rect=(0, 0.01, 1, 0.97))
out = os.path.join(config.A1_DIR, "cisc330_a1_g1", "figures", "task6_cursor_compare.png")
fig.savefig(out, dpi=130); print("wrote", out)
