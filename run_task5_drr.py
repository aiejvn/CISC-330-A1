"""
run_task5_drr.py - Task 5 (sphere phase): render the DRRs of both poses and save them.

Saves outputs/drr_sphere_A.npy, drr_sphere_B.npy (float I/I0, shape (n, n), indexed [j, k])
and outputs/task5_drr_sphere.png. Task 6 reads the .npy files, not the PNG.
PIXEL_MM = 0.1 is the full resolution (2000 x 2000); use 1.0 for a quick coarse picture.

One image per detector. Each image is computed from that detector's OWN geometry:
  drr.render_drr(pose, ...) calls
    frames.detector_geometry(pose)  -> source S, detector center C, axes u, v, w  (CK frame)
    frames.F_CK_from_Det(pose)      -> maps every pixel middle (u, v, 0) of that detector to CK
  and shoots one ray from S through every pixel middle. Nothing is shared between the poses.
    pose A (+45 deg about z): S = (-707, 707, 0), C = ( 707, -707, 0), v = ( 0.707,  0.707, 0)
    pose B (-45 deg about z): S = ( 707, 707, 0), C = (-707, -707, 0), v = ( 0.707, -0.707, 0)
  (u is CK +z for both.) In both poses the central ray S -> C goes through the CK origin,
  i.e. through the sphere center, so the sphere sits in the middle of both images; the
  three steel balls, which are at different places on the two rays, land at different
  pixels in A and in B (e.g. M1 near pixel row 876 in A, row 1660 in B), and the two
  arrays are not equal (max |A - B| ~ 0.22).

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config, drr
from markers import load_markers
from projector import project_to_detector

PIXEL_MM = 0.1

M_CK = load_markers()["CK"]
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, pose in zip(axes, ("A", "B")):
    # Pose "A" -> detector A's source and detector plane; pose "B" -> detector B's. The same
    # sphere and marker positions (CK frame) are seen from two different directions.
    img = drr.render_drr(pose, PIXEL_MM, M_CK)
    np.save(os.path.join(config.OUTPUT_DIR, f"drr_sphere_{pose}.npy"), img)
    print(f"pose {pose}: shape {img.shape}, min intensity {img.min():.5f}, max intensity {img.max():.5f}")
    ax.imshow(img, origin="lower", cmap="gray", extent=[-100, 100, -100, 100])   # origin lower: image frame origin = lower-left
    # Text-only marker names (M1, M2, M3) next to each ball's shadow; the position comes from
    # the projector, which is the same geometry the rays above use. No symbols are drawn.
    for name, (u, v) in zip(config.MARKER_NAMES, project_to_detector(M_CK, pose)):
        ax.annotate(name, (u, v), color="yellow", xytext=(8, 8), textcoords="offset points")
    ax.set_title(f"DRR detector {pose} (sphere + 3 markers)")
    ax.set_xlabel("u (mm)  = image i1")
    ax.set_ylabel("v (mm)  = image i2")
fig.tight_layout()
fig.savefig(os.path.join(config.OUTPUT_DIR, "task5_drr_sphere.png"), dpi=120)
print("saved outputs/drr_sphere_A.npy, drr_sphere_B.npy, task5_drr_sphere.png")
