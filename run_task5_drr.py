"""
run_task5_drr.py - Task 5: render the DRRs of both poses, first the sphere phase, then the vertebra phase.
PIXEL_MM = 0.1 is the full resolution (2000 x 2000); use 1.0 for a quick coarse picture.

Sphere phase (sphere R = 50 mm + three steel balls at the table positions)
  saves outputs/drr_sphere_A.npy, drr_sphere_B.npy (float sum d_i x_i in g/cm^2, shape (n, n), indexed [j, k])
  and outputs/task5_drr_sphere.png. Task 6 reads the .npy files, not the PNG.

  One image per detector. Each image is computed from that detector's OWN geometry:
    drr.render_sphere_drr(pose, ...) calls
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

Vertebra phase (STL vertebra with its own 2 mm marker balls)
  saves outputs/drr_vertebra_A.npy, drr_vertebra_B.npy   float sum d_i x_i in g/cm^2, shape (n, n), indexed [j, k]
        outputs/markers_stl_projected.npz                TRUE marker data of this DRR (Task 6 ground truth):
            CK (3, 3) STL ball centres in CK, DetA_uv_mm / DetB_uv_mm (3, 2), ImgA_px / ImgB_px (3, 2)
            (the table-based projections of Task 4 differ from these by 0.35 - 0.72 mm)
        outputs/task5_drr_vertebra.png                   picture (sum d_i x_i) with the true centres marked

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config, frames, drr
from markers import load_markers
from stl_io import read_binary_stl, split_vertebra_and_markers
from geometry import apply_transform
from projector import project_to_detector, detector_to_image

PIXEL_MM = 0.1


def run_sphere():
    M_CK = load_markers()["CK"]          # marker positions of the table (Task 2), in the CK frame
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, pose in zip(axes, ("A", "B")):
        # Pose "A" -> detector A's source and detector plane; pose "B" -> detector B's. The same
        # sphere and marker positions (CK frame) are seen from two different directions.
        img = drr.render_sphere_drr(pose, PIXEL_MM, M_CK)
        np.save(os.path.join(config.OUTPUT_DIR, f"drr_sphere_{pose}.npy"), img)
        print(f"sphere pose {pose}: shape {img.shape}, min {img.min():.5f}, max {img.max():.5f} g/cm^2")
        # dark = more material along the ray (larger sum d x); origin lower so the picture matches the image frame
        ax.imshow(img, origin="lower", cmap="gray_r", extent=[-100, 100, -100, 100])   # origin lower: image frame origin = lower-left
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


def run_vertebra():
    # the STL holds the bone and the three marker balls as separate closed pieces; the balls are 2 mm and sit
    # 0.3-0.4 mm from the table, so we render the STL balls and keep their true centers as ground truth for Task 6
    triangles = read_binary_stl(config.STL_PATH)
    bone_MD, balls_MD, centers_MD = split_vertebra_and_markers(triangles, config.MARKERS_MD_mm)
    F = frames.F_CK_from_MD()
    to_CK = lambda t: apply_transform(F, t.reshape(-1, 3)).reshape(-1, 3, 3)
    bone, balls = to_CK(bone_MD), [to_CK(b) for b in balls_MD]
    centers_CK = apply_transform(F, centers_MD)

    truth = {"CK": centers_CK}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, pose in zip(axes, ("A", "B")):
        img, info = drr.render_vertebra_drr(pose, PIXEL_MM, bone, balls)
        np.save(os.path.join(config.OUTPUT_DIR, f"drr_vertebra_{pose}.npy"), img)
        uv = project_to_detector(centers_CK, pose)       # where the ball centers really land (not the table values)
        truth[f"Det{pose}_uv_mm"] = uv
        truth[f"Img{pose}_px"] = detector_to_image(uv)
        print(f"vertebra pose {pose}: min {img.min():.3f} max {img.max():.3f} g/cm^2; "
              f"entry/exit balance nonzero pixels: bone {np.count_nonzero(info['bal_bone'])}, balls {np.count_nonzero(info['bal_balls'])}")
        ax.imshow(img, origin="lower", cmap="gray_r", extent=[-100, 100, -100, 100])
        ax.plot(uv[:, 0], uv[:, 1], "r+", ms=14, mew=1)
        for name, (u, v) in zip(config.MARKER_NAMES, uv):
            ax.annotate(name, (u, v), color="r", xytext=(6, 6), textcoords="offset points")
        ax.set_title(f"DRR pose {pose}: sum d_i x_i (g/cm^2); red + = STL ball centres")
        ax.set_xlabel("u (mm) = image i1"); ax.set_ylabel("v (mm) = image i2")
    np.savez(os.path.join(config.OUTPUT_DIR, "markers_stl_projected.npz"), **truth)
    fig.tight_layout()
    fig.savefig(os.path.join(config.OUTPUT_DIR, "task5_drr_vertebra.png"), dpi=110)
    print("saved outputs/drr_vertebra_A.npy, drr_vertebra_B.npy, markers_stl_projected.npz, task5_drr_vertebra.png")


if __name__ == "__main__":
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    run_sphere()
    run_vertebra()
