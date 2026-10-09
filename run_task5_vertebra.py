"""
run_task5_vertebra.py - Task 5 (vertebra phase): DRRs of the STL vertebra with its own
2 mm marker balls, both poses, full resolution (PIXEL_MM = 0.1 -> 2000 x 2000).

Saves
  outputs/drr_vertebra_A.npy, drr_vertebra_B.npy   float I/I0, shape (n, n), indexed [j, k]
  outputs/markers_stl_projected.npz                TRUE marker data of this DRR (Task 6 ground truth):
        CK (3, 3) STL ball centres in CK, DetA_uv_mm / DetB_uv_mm (3, 2), ImgA_px / ImgB_px (3, 2)
        (the table-based projections of Task 4 differ from these by 0.35 - 0.72 mm)
  outputs/task5_drr_vertebra.png                   picture (absorbance -ln(I/I0)) with the true centres marked

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config, frames, drr
from stl_io import read_binary_stl, split_vertebra_and_markers
from geometry import apply_transform
from projector import project_to_detector, detector_to_image

PIXEL_MM = 0.1

triangles = read_binary_stl(config.STL_PATH)
bone_MD, balls_MD, centers_MD = split_vertebra_and_markers(triangles, config.MARKERS_MD_mm)
F = frames.F_CK_from_MD()
to_CK = lambda t: apply_transform(F, t.reshape(-1, 3)).reshape(-1, 3, 3)
bone, balls = to_CK(bone_MD), [to_CK(b) for b in balls_MD]
centers_CK = apply_transform(F, centers_MD)

truth = {"CK": centers_CK}
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
for ax, pose in zip(axes, ("A", "B")):
    img, info = drr.render_vertebra_drr(pose, PIXEL_MM, bone, balls)
    np.save(os.path.join(config.OUTPUT_DIR, f"drr_vertebra_{pose}.npy"), img)
    uv = project_to_detector(centers_CK, pose)
    truth[f"Det{pose}_uv_mm"] = uv
    truth[f"Img{pose}_px"] = detector_to_image(uv)
    print(f"pose {pose}: I/I0 min {img.min():.3e} max {img.max():.3f}; "
          f"entry/exit balance nonzero pixels: bone {np.count_nonzero(info['bal_bone'])}, balls {np.count_nonzero(info['bal_balls'])}")
    ax.imshow(-np.log(img), origin="lower", cmap="gray_r", extent=[-100, 100, -100, 100])
    ax.plot(uv[:, 0], uv[:, 1], "r+", ms=14, mew=1)
    for name, (u, v) in zip(config.MARKER_NAMES, uv):
        ax.annotate(name, (u, v), color="r", xytext=(6, 6), textcoords="offset points")
    ax.set_title(f"DRR pose {pose}: absorbance -ln(I/I0); red + = STL ball centres")
    ax.set_xlabel("u (mm) = image i1"); ax.set_ylabel("v (mm) = image i2")
np.savez(os.path.join(config.OUTPUT_DIR, "markers_stl_projected.npz"), **truth)
fig.tight_layout()
fig.savefig(os.path.join(config.OUTPUT_DIR, "task5_drr_vertebra.png"), dpi=110)
print("saved outputs/drr_vertebra_A.npy, drr_vertebra_B.npy, markers_stl_projected.npz, task5_drr_vertebra.png")
