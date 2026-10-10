"""
export_visual_crops.py - gridded zoom crops of every marker in the four DRRs, for reading the
marker centers by eye (Task 6 "visual localization").

The crop is centered on a coarse position (the algorithm's result rounded to a multiple of 50 px,
so the crop does not give the answer away) and does NOT draw the found or the true center.
Grid lines every 10 px, labelled with the absolute image pixel (i1 across, i2 up).

Run from the A1 folder:   python tools/export_visual_crops.py
Writes outputs/visual/crops_<phase>_<pose>.png
"""
import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

HALF = 60   # crop is 120 x 120 px (12 mm)
out = os.path.join(config.OUTPUT_DIR, "visual"); os.makedirs(out, exist_ok=True)
found = np.load(os.path.join(config.OUTPUT_DIR, "markers_localized_task6.npz"))
for phase in ("sphere", "vertebra"):
    for pose in ("A", "B"):
        img = np.load(os.path.join(config.OUTPUT_DIR, f"drr_{phase}_{pose}.npy"))
        pts = found[f"{phase}_{pose}_px"]
        order = np.argsort(pts[:, 0])
        fig, axs = plt.subplots(1, 3, figsize=(18, 6.4))
        for ax, p in zip(axs, pts[order]):
            k0, j0 = (int(round(p[0] / 50.0)) * 50, int(round(p[1] / 50.0)) * 50)
            win = img[j0 - HALF:j0 + HALF, k0 - HALF:k0 + HALF]
            ax.imshow(win, cmap="gray_r", origin="lower", interpolation="nearest",
                      extent=[k0 - HALF, k0 + HALF, j0 - HALF, j0 + HALF])
            ticks_k = np.arange(k0 - HALF, k0 + HALF + 1, 10); ticks_j = np.arange(j0 - HALF, j0 + HALF + 1, 10)
            ax.set_xticks(ticks_k); ax.set_yticks(ticks_j)
            ax.grid(True, color="tab:blue", lw=0.6, alpha=0.7)
            ax.tick_params(labelsize=8)
            ax.set_xlabel("i1 (px)"); ax.set_ylabel("i2 (px)")
            ax.set_title(f"{phase} {pose}: crop centred at ({k0}, {j0})")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"crops_{phase}_{pose}.png"), dpi=70); plt.close(fig)
print("wrote", out)
