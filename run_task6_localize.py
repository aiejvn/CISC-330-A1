"""
run_task6_localize.py - Task 6: find the marker centers in the saved DRRs.

Reads outputs/drr_sphere_{A,B}.npy and outputs/drr_vertebra_{A,B}.npy (Task 5), runs
localize.localize_markers on each and saves
  outputs/markers_localized_task6.npz   keys  <phase>_<pose>_px  and  <phase>_<pose>_uv_mm,
        phase in {sphere, vertebra}, pose in {A, B}; shape (3, 2); order = best score first
        (the markers are identical, so the order says nothing about M1, M2, M3: that is Task 8)
  outputs/task6_localization.png         the DRRs with the found centers (red +), the true centers (green o)
                                         and the error in px / mm, plus a zoom on the best marker
The sub-pixel "centroid" result is the one used by later tasks.

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import config
from localize import localize_markers, image_px_to_detector_mm

TRUTH = {"sphere": "markers_projected_task4.npz", "vertebra": "markers_stl_projected.npz"}   # same as the test
saved = {}
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for row, phase in enumerate(("sphere", "vertebra")):
    truth = np.load(os.path.join(config.OUTPUT_DIR, TRUTH[phase]))
    for col, pose in enumerate(("A", "B")):
        img = np.load(os.path.join(config.OUTPUT_DIR, f"drr_{phase}_{pose}.npy"))
        res = localize_markers(img)
        px = res["centroid_px"]          # the sub-pixel, weighted center of step 2 is what later tasks use
        saved[f"{phase}_{pose}_px"] = px
        saved[f"{phase}_{pose}_uv_mm"] = image_px_to_detector_mm(px)   # image frame (pixels) -> detector frame (mm) for Task 7
        # the found markers have no names, so each is compared with the nearest true marker
        true_px = truth[f"Img{pose}_px"]
        dist = np.linalg.norm(px[:, None, :] - true_px[None, :, :], axis=2)     # dist[found, true] in pixels
        match = true_px[dist.argmin(axis=1)]                                     # true center belonging to each found marker
        err = dist.min(axis=1)                                                   # distance found -> true, pixels
        print(f"{phase} {pose}: scores {np.round(res['score'], 3)}, next candidate {res['next_score']:.3f}")
        for p, q, e in zip(res["peak_px"], px, err):
            print(f"   peak (i1, i2) = {p}   centroid = {np.round(q, 2)}   (u, v) = {np.round(image_px_to_detector_mm(q), 3)} mm"
                  f"   error to true = {e:.3f} px = {e * config.PIXEL_SIZE_mm:.4f} mm")
        ax = axes[row, 2 * col]
        ax.imshow(img, origin="lower", cmap="gray_r", extent=[0, img.shape[1], 0, img.shape[0]])
        ax.plot(match[:, 0], match[:, 1], "o", mfc="none", mec="limegreen", ms=18, mew=1.5)   # true centers
        ax.plot(px[:, 0], px[:, 1], "r+", ms=14, mew=1.5)                                     # found centers
        ax.set_title(f"{phase} DRR, detector {pose}: found (red +) vs true (green o)\n"
                     f"largest error {err.max():.2f} px = {err.max() * config.PIXEL_SIZE_mm:.3f} mm")
        ax.set_xlabel("i1 (px)"); ax.set_ylabel("i2 (px)")
        ax = axes[row, 2 * col + 1]                                    # zoom on the strongest marker
        k, j = int(px[0, 0]), int(px[0, 1])
        ax.imshow(img[j - 60:j + 61, k - 60:k + 61], origin="lower", cmap="gray_r",
                  extent=[k - 60, k + 61, j - 60, j + 61])
        ax.plot(match[0, 0], match[0, 1], "o", mfc="none", mec="limegreen", ms=22, mew=1.5)
        ax.plot(px[0, 0], px[0, 1], "r+", ms=16, mew=1.5)
        ax.plot([px[0, 0], match[0, 0]], [px[0, 1], match[0, 1]], "-", color="orange", lw=1)   # the error, drawn as a line
        ax.set_title(f"zoom on the best marker ({phase} {pose})\n"
                     f"error {err[0]:.2f} px = {err[0] * config.PIXEL_SIZE_mm:.3f} mm")
fig.tight_layout(rect=(0, 0.04, 1, 1))                  # leave a strip at the bottom for the legend
# one legend for the whole figure (the same symbols are used in every panel)
fig.legend(handles=[Line2D([], [], marker="+", color="r", ls="", ms=12, mew=1.5, label="found center"),
                    Line2D([], [], marker="o", mfc="none", mec="limegreen", ls="", ms=12, mew=1.5, label="true center"),
                    Line2D([], [], color="orange", lw=1, label="error (found to true, zoom panels)")],
           loc="lower center", ncol=3, fontsize=11, frameon=False)
fig.savefig(os.path.join(config.OUTPUT_DIR, "task6_localization.png"), dpi=100)
np.savez(os.path.join(config.OUTPUT_DIR, "markers_localized_task6.npz"), **saved)
print("saved outputs/markers_localized_task6.npz, task6_localization.png")
