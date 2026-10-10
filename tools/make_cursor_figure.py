"""
make_cursor_figure.py - report figure for Task 6 "visual localization": what the picker (tools/visual_picker.py)
shows when the cursor is on a marker. Left: whole DRR with the cursor crosshair and the status-bar text.
Right: zoom on that marker with the cursor, the visual reading, the algorithm's center and the true center.
Uses the readings in outputs/task6_visual_readings.csv. Writes cisc330_a1_g1/figures/task6_cursor_view.png.
Run from the A1 folder:   python tools/make_cursor_figure.py [phase pose marker_index]
"""
import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

phase, pose, idx = (sys.argv[1], sys.argv[2], int(sys.argv[3])) if len(sys.argv) == 4 else ("vertebra", "B", 1)
rows = [r for r in csv.DictReader(open(os.path.join(config.OUTPUT_DIR, "task6_visual_readings.csv")))
        if r["phase"] == phase and r["pose"] == pose]
r = rows[idx]
vis = np.array([float(r["visual_i1"]), float(r["visual_i2"])])
alg = np.array([float(r["algo_i1"]), float(r["algo_i2"])])
tru = np.array([float(r["true_i1"]), float(r["true_i2"])])
img = np.load(os.path.join(config.OUTPUT_DIR, f"drr_{phase}_{pose}.npy"))
n, px = img.shape[0], config.PIXEL_SIZE_mm
status = (f"i1={vis[0]:7.1f}  i2={vis[1]:7.1f} px  |  u={vis[0] * px - 100:6.2f}  v={vis[1] * px - 100:6.2f} mm"
          f"  |  sum d x = {img[int(vis[1]), int(vis[0])]:.3f} g/cm^2")
H = 40
fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1, 1]})
a0.imshow(img, cmap="gray_r", origin="lower", extent=[0, n, 0, n], interpolation="nearest")
a0.axvline(vis[0], color="tab:blue", lw=0.8); a0.axhline(vis[1], color="tab:blue", lw=0.8)
a0.add_patch(Rectangle((vis[0] - H, vis[1] - H), 2 * H, 2 * H, fill=False, ec="orange", lw=1.5))
a0.set_xlabel("i1 (px)"); a0.set_ylabel("i2 (px)")
a0.set_title(f"(a) {phase} DRR, detector {pose}: cursor on a marker")
a0.text(0.5, -0.17, status, transform=a0.transAxes, ha="center", va="top", fontsize=8, family="monospace",
        bbox=dict(boxstyle="square", fc="#eeeeee", ec="grey"))
a1.imshow(img, cmap="gray_r", origin="lower", extent=[0, n, 0, n], interpolation="nearest")
a1.set_xlim(vis[0] - H, vis[0] + H); a1.set_ylim(vis[1] - H, vis[1] + H)
a1.set_xticks(np.arange(int(vis[0]) - H, int(vis[0]) + H + 1, 10)); a1.set_yticks(np.arange(int(vis[1]) - H, int(vis[1]) + H + 1, 10))
a1.grid(True, color="tab:blue", lw=0.5, alpha=0.6)
a1.axvline(vis[0], color="tab:blue", lw=1.0); a1.axhline(vis[1], color="tab:blue", lw=1.0)
a1.plot(*vis, "x", color="tab:blue", ms=14, mew=2, label=f"visual reading ({vis[0]:.0f}, {vis[1]:.0f})")
a1.plot(*alg, "+", color="red", ms=18, mew=1.5, label=f"algorithm ({alg[0]:.1f}, {alg[1]:.1f})")
a1.plot(*tru, "o", mfc="none", mec="limegreen", ms=22, mew=1.5, label=f"true projection ({tru[0]:.1f}, {tru[1]:.1f})")
a1.legend(loc="upper right", fontsize=8, framealpha=0.9)
a1.set_xlabel("i1 (px)"); a1.set_ylabel("i2 (px)")
a1.set_title(f"(b) zoom: visual {float(r['visual_vs_true_px']):.2f} px, algorithm {float(r['algo_vs_true_px']):.2f} px from true")
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = os.path.join(config.A1_DIR, "cisc330_a1_g1", "figures", "task6_cursor_view.png")
fig.savefig(out, dpi=150); print("wrote", out)
