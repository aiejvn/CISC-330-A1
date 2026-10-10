"""
window_sweep.py - Task 6: worst centroid error (px) against the true projections as a function of the
refinement window size. Writes outputs/task6_window_sweep.txt (the numbers quoted in the report).
The window was tuned on these same images; the sphere (clean background) and the vertebra (uneven
background) behave the same way, so the choice of 81 px is not an artefact of one object.
Run from the A1 folder:   python tools/window_sweep.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, localize

TRUTH = {"sphere": "markers_projected_task4.npz", "vertebra": "markers_stl_projected.npz"}
imgs = {(p, s): np.load(os.path.join(config.OUTPUT_DIR, f"drr_{p}_{s}.npy")) for p in TRUTH for s in "AB"}
true = {(p, s): np.load(os.path.join(config.OUTPUT_DIR, TRUTH[p]))[f"Img{s}_px"] for p in TRUTH for s in "AB"}
lines = ["window (px) | worst centroid error, sphere (px) | vertebra (px)"]
for half in (30, 35, 40, 45, 50, 60, 75):
    localize.WINDOW_HALF_PX = half
    worst = {}
    for p in TRUTH:
        w = 0.0
        for s in "AB":
            r = localize.localize_markers(imgs[(p, s)])["centroid_px"]
            w = max(w, np.linalg.norm(r[:, None] - true[(p, s)][None], axis=2).min(axis=1).max())
        worst[p] = w
    lines.append(f"{2 * half + 1:11d} | {worst['sphere']:34.2f} | {worst['vertebra']:.2f}")
text = "\n".join(lines); print(text)
open(os.path.join(config.OUTPUT_DIR, "task6_window_sweep.txt"), "w").write(text + "\n")
