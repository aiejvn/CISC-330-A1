"""
Task 5 qualitative check: LOOK at the saved DRRs first, then compare with what we expected.
Reads outputs/drr_sphere_{A,B}.npy and outputs/drr_vertebra_{A,B}.npy (run run_task5_drr.py first).

What we expect to SEE (written before running):
 1. Every marker is a small dark round dot (steel absorbs more than sphere / bone).
 2. The dot is about 3 mm x magnification across, magnification = SDD / (source -> marker distance
    along the beam axis), so about 6 mm = 60 pixels.
 3. The dot sits where Task 4 (sphere) or the STL ball center (vertebra) says it should.
 4. Sphere: the dot is clearly visible on a smooth background. Vertebra: dots on or near the bone,
    partly hidden by it; they may be harder to see by eye.

For each marker the script prints (and saves one picture per object, pose A, outputs/task5_qualitative.png): does a dot exist, how far is its center from the expected
point, how wide is it compared with the expected width.

Run from the A1 folder:   python tests/test_task5_qualitative.py

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, frames
from markers import load_markers
from projector import project_to_detector

PX = config.PIXEL_SIZE_mm          # the saved images are full resolution (0.1 mm)
HALF = 12.0                        # half width of the picture window (mm)
SEARCH = 3.0                       # look for the dot within this distance of the expected point (mm)
RING = (4.5, 8.0)                  # background = plane fitted to the ring between these distances (mm), outside the dot


def magnification(pose, M):
    det = frames.detector_geometry(pose)
    return config.SDD_mm / ((det["S_CK"] - M) @ det["w"])


def to_px(u, v):
    return (u + 100) / PX - 0.5, (v + 100) / PX - 0.5          # column k, row j (pixel middles)


def analyse(img, uv, diameter_mm):
    """Dot = image minus its blurred copy. Returns (found center uv, width mm, contrast)."""
    k0, j0 = to_px(*uv)
    r = int(HALF / PX)
    k0, j0 = to_px(*uv)
    k0, j0 = int(round(k0)), int(round(j0))
    win = img[j0 - r: j0 + r, k0 - r: k0 + r]
    kk, jj = np.meshgrid(np.arange(-r, r), np.arange(-r, r))
    dist = np.hypot(kk, jj) * PX
    ring = (dist >= RING[0]) & (dist <= RING[1])
    A = np.c_[np.ones(ring.sum()), kk[ring], jj[ring]]
    coef = np.linalg.lstsq(A, win[ring], rcond=None)[0]
    dot = win - (coef[0] + coef[1] * kk + coef[2] * jj)                  # dark = positive
    near = dist <= SEARCH
    jp, kp = np.unravel_index(np.argmax(np.where(near, dot, -np.inf)), dot.shape)
    peak = dot[jp, kp]
    # width: the dot cut at 25 % of its peak (a chord profile reaches 25 % at 0.97 of the diameter)
    mask = dot > 0.25 * peak
    row = mask[jp]
    left = kp
    while left > 0 and row[left - 1]: left -= 1
    right = kp
    while right < len(row) - 1 and row[right + 1]: right += 1
    width = (right - left + 1) * PX
    found = np.array([uv[0] + (kp - r) * PX, uv[1] + (jp - r) * PX])
    return found, width, peak, dot, win


report = []
phases = [("sphere", load_markers()["CK"]),
          ("vertebra", np.load(os.path.join(config.OUTPUT_DIR, "markers_stl_projected.npz"))["CK"])]
report.append("Task 5 qualitative check (what the markers look like, and how close they are to what we expected)\n")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for phase, M_CK in phases:
    for pose in ("A", "B"):
        img = np.load(os.path.join(config.OUTPUT_DIR, f"drr_{phase}_{pose}.npy"))
        uv_all = project_to_detector(M_CK, pose)
        found_all = []
        for name, M, uv in zip(config.MARKER_NAMES, M_CK, uv_all):
            mag = magnification(pose, M)
            expect_w = config.MARKER_DIAMETER_mm * mag
            found, width, peak, dot, win = analyse(img, uv, expect_w)
            found_all.append((found, width))
            off = np.linalg.norm(found - uv)
            visible = peak > 0.5                      # a steel ball adds up to ~2.3 g/cm^2
            report.append(f"{phase:8s} pose {pose} {name}: dark dot {'FOUND' if visible else 'NOT FOUND'}; "
                          f"strength {peak:.2f} g/cm^2 above its surroundings; "
                          f"center {off:.2f} mm from the expected point; "
                          f"width {width:.1f} mm (expected {expect_w:.1f} mm, magnification {mag:.2f})")
        if pose == "A":                               # picture: pose A, expected vs calculated
            ax = axes[0 if phase == "sphere" else 1]
            lo = uv_all.min(0) - 12
            hi = uv_all.max(0) + 12
            k0, j0 = [int(round(x)) for x in to_px(*lo)]
            k1, j1 = [int(round(x)) for x in to_px(*hi)]
            ax.imshow(img[j0:j1, k0:k1], origin="lower", cmap="gray_r",
                      extent=[lo[0], hi[0], lo[1], hi[1]])
            for i, (uv, (found, width)) in enumerate(zip(uv_all, found_all)):
                ew = config.MARKER_DIAMETER_mm * magnification(pose, M_CK[i])
                ax.add_patch(plt.Circle(uv, ew / 2, fill=False, color="red", lw=1.2,
                                        label="expected (Task 4 / STL center, 3 mm x magnification)" if i == 0 else None))
                ax.add_patch(plt.Circle(found, width / 2, fill=False, color="cyan", lw=1.2, ls="--",
                                        label="calculated (measured in the DRR)" if i == 0 else None))
                ax.annotate(config.MARKER_NAMES[i], uv, color="yellow", xytext=(12, 12), textcoords="offset points")
            ax.set_title(f"{phase}, detector A")
            ax.set_xlabel("u (mm)"); ax.set_ylabel("v (mm)")
            ax.legend(loc="center", fontsize=8)
fig.suptitle("Task 5: expected vs calculated marker (dark = larger sum d x)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(config.OUTPUT_DIR, "task5_qualitative.png"), dpi=110)

text = "\n".join(report)
print(text)
with open(os.path.join(config.OUTPUT_DIR, "task5_qualitative.txt"), "w") as f:
    f.write(text + "\n")
