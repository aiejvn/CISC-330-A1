"""
run_task6_localize.py - Task 6: find the marker centers in the saved DRRs.

Reads outputs/drr_sphere_{A,B}.npy and outputs/drr_vertebra_{A,B}.npy (Task 5), runs
localize.localize_markers on each and saves
  outputs/markers_localized_task6.npz   keys  <phase>_<pose>_px  and  <phase>_<pose>_uv_mm,
        phase in {sphere, vertebra}, pose in {A, B}; shape (3, 2); order = best score first
        (the markers are identical, so the order says nothing about M1, M2, M3: that is Task 8)
  outputs/task6_localization.png         the DRRs with the found centers and a zoom on every marker
The sub-pixel "centroid" result is the one used by later tasks.

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from localize import localize_markers, image_px_to_detector_mm

saved = {}
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for row, phase in enumerate(("sphere", "vertebra")):
    for col, pose in enumerate(("A", "B")):
        img = np.load(os.path.join(config.OUTPUT_DIR, f"drr_{phase}_{pose}.npy"))
        res = localize_markers(img)
        px = res["centroid_px"]          # the sub-pixel, weighted center of step 2 is what later tasks use
        saved[f"{phase}_{pose}_px"] = px
        saved[f"{phase}_{pose}_uv_mm"] = image_px_to_detector_mm(px)   # image frame (pixels) -> detector frame (mm) for Task 7
        print(f"{phase} {pose}: scores {np.round(res['score'], 3)}, next candidate {res['next_score']:.3f}")
        for p, q in zip(res["peak_px"], px):
            print(f"   peak (i1, i2) = {p}   centroid = {np.round(q, 2)}   (u, v) = {np.round(image_px_to_detector_mm(q), 3)} mm")
        ax = axes[row, 2 * col]
        ax.imshow(img, origin="lower", cmap="gray_r", extent=[0, img.shape[1], 0, img.shape[0]])
        ax.plot(px[:, 0], px[:, 1], "r+", ms=14, mew=1.5)
        ax.set_title(f"{phase} DRR, detector {pose}: found centers (red +)")
        ax.set_xlabel("i1 (px)"); ax.set_ylabel("i2 (px)")
        ax = axes[row, 2 * col + 1]                                    # zoom on the strongest marker
        k, j = int(px[0, 0]), int(px[0, 1])
        ax.imshow(img[j - 60:j + 61, k - 60:k + 61], origin="lower", cmap="gray_r",
                  extent=[k - 60, k + 61, j - 60, j + 61])
        ax.plot(px[0, 0], px[0, 1], "r+", ms=16, mew=1.5)
        ax.set_title(f"zoom on the best marker ({phase} {pose})")
fig.tight_layout()
fig.savefig(os.path.join(config.OUTPUT_DIR, "task6_localization.png"), dpi=100)
np.savez(os.path.join(config.OUTPUT_DIR, "markers_localized_task6.npz"), **saved)
print("saved outputs/markers_localized_task6.npz, task6_localization.png")
