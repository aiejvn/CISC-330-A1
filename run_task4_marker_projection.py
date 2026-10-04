"""
Task 4 runner (Marker Projector): project the three markers from the CK frame
onto both detectors, also transform them to the corresponding image frames,
and save the results for further use.

Test (assignment): "for each marker, based on a 3D sketch, estimate with a hand
calculation where they should 'roughly' project and show that they indeed did."
Our sketch is part (a) of the Task 2 figure (outputs/task2_markers_in_CK.png).
Hand calculation: the CK center is SAD = 1000 mm from the source and the
detector is SDD = 2000 mm from the source, so the CK center is halfway between
them. All markers are within about 50 mm of the CK center, so by similar
triangles each marker's offset from the beam central line should roughly
double on the detector. For a marker at (x, y, z) in the CK frame:
      u ~ 2 z                       (the detector's +u is the CK's +z)
      v ~ 2 cos45 (x + y)     on A  (v_A = (cos45, sin45, 0))
      v ~ 2 cos45 (x - y)     on B  (v_B = (cos45, -sin45, 0))

Run from the A1 folder (Mac Terminal):   python3 run_task4_marker_projection.py

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12})   # readable when the figure is shrunk to page width
import config, frames
from markers import load_markers
from projector import project_to_detector, detector_to_image, on_detector
from geometry import apply_transform
from stl_io import read_binary_stl, corner_points

markers = load_markers()                   # the markers saved by Task 2 (all frames)
M_CK = markers["CK"]                       # the three markers in the CK frame, shape (3, 3)
result = {}                                # what we save for later tasks
lines = [f"MD->CK mode: {config.MD_TO_CK_MODE}", "",
         "marker det | hand calc. (u, v) mm   | projector (u, v) mm  | image (i1, i2) px  | on detector"]
largest_difference = 0.0
for p in ("A", "B"):
    uv = project_to_detector(M_CK, p)                 # Forward Projector (Task 3), detector frame, mm
    px = detector_to_image(uv)                        # detector frame -> image frame, pixels
    result[f"Det{p}_uv_mm"], result[f"Img{p}_px"] = uv, px
    x, y, z = M_CK[:, 0], M_CK[:, 1], M_CK[:, 2]
    cos45 = np.cos(np.radians(45.0))
    if p == "A":
        hand = np.stack([2 * z, 2 * cos45 * (x + y)], axis=1)   # hand calculation for detector A
    else:
        hand = np.stack([2 * z, 2 * cos45 * (x - y)], axis=1)   # hand calculation for detector B
    result[f"Det{p}_hand_uv_mm"] = hand
    # how far apart the hand calculation and the projector are on the detector (distance in mm)
    largest_difference = max(largest_difference, np.linalg.norm(hand - uv, axis=1).max())
    for i, n in enumerate(config.MARKER_NAMES):
        lines.append(f"{n:6s} {p:3s} | ({hand[i,0]:8.3f}, {hand[i,1]:8.3f})     | "
                     f"({uv[i,0]:8.3f}, {uv[i,1]:8.3f}) | ({px[i,0]:9.3f}, {px[i,1]:9.3f}) | {bool(on_detector(uv[i]))}")
lines.append(f"\nlargest distance between the hand calculation and the projector: {largest_difference:.3f} mm")
text = "\n".join(lines)
print(text)
with open(os.path.join(config.OUTPUT_DIR, "task4_marker_projection.txt"), "w") as f:
    f.write(text + "\n")

# ---- save for later tasks (.npz for code, .csv for people) ----
np.savez(os.path.join(config.OUTPUT_DIR, "markers_projected_task4.npz"),
         **{k: result[k] for k in ("DetA_uv_mm", "ImgA_px", "DetB_uv_mm", "ImgB_px")})
with open(os.path.join(config.OUTPUT_DIR, "markers_projected_task4.csv"), "w") as f:
    f.write("marker,detector,u_mm,v_mm,i1_px,i2_px\n")
    for p in ("A", "B"):
        for i, n in enumerate(config.MARKER_NAMES):
            u, v = result[f"Det{p}_uv_mm"][i]
            i1, i2 = result[f"Img{p}_px"][i]
            f.write(f"{n},{p},{u:.4f},{v:.4f},{i1:.2f},{i2:.2f}\n")

# ---- picture: the markers on each detector, over the projected vertebra model ----
# Axes: u to the right and v up, so w points out of the page (we look from the
# source side). The bottom/left axes are the detector frame in mm; the top/right
# axes are the image frame in pixels (i = 10 u + 1000), whose (0, 0) is the
# lower left corner of the detector.
vertebra_CK = apply_transform(frames.F_CK_from_MD(), corner_points(read_binary_stl(config.STL_PATH)))
colors = {"M1": "tab:red", "M2": "tab:green", "M3": "tab:blue"}
fig, axs = plt.subplots(1, 2, figsize=(11, 5.9))
for ax, p in zip(axs, ("A", "B")):
    # every corner point of the STL surface, projected; there are so many that they merge into a grey shape
    vertebra_uv = project_to_detector(vertebra_CK, p)
    ax.scatter(vertebra_uv[:, 0], vertebra_uv[:, 1], s=0.05, c="0.7", rasterized=True)
    for i, n in enumerate(config.MARKER_NAMES):
        u, v = result[f"Det{p}_uv_mm"][i]
        hand_u, hand_v = result[f"Det{p}_hand_uv_mm"][i]
        ax.plot(u, v, "o", color=colors[n], ms=11, mfc="none", mew=2.2)
        ax.plot(hand_u, hand_v, "x", color=colors[n], ms=9, mew=2)
        ax.annotate(n, (u, v), xytext=(10, 6), textcoords="offset points",
                    color=colors[n], fontsize=13, fontweight="bold")
    ax.plot(0, 0, "k+", ms=12, mew=1.5)                 # detector center = image center
    ax.set_xlim(-100, 100); ax.set_ylim(-100, 100)
    ax.set_xticks(np.arange(-100, 101, 25)); ax.set_yticks(np.arange(-100, 101, 25))
    ax.set_aspect("equal"); ax.grid(alpha=0.35)
    ax.set_xlabel("$u$ (mm)"); ax.set_ylabel("$v$ (mm)")
    ax.set_title(f"Detector {p}", pad=10)
    to_px = (lambda x: 10 * x + 1000, lambda x: (x - 1000) / 10)   # mm -> px and back
    top = ax.secondary_xaxis("top", functions=to_px); top.set_xlabel("$i_1$ (pixels)")
    right = ax.secondary_yaxis("right", functions=to_px); right.set_ylabel("$i_2$ (pixels)")
    top.set_xticks(np.arange(0, 2001, 500)); right.set_yticks(np.arange(0, 2001, 500))
# one legend for both panels
handles = [plt.Line2D([], [], marker="o", ls="", color="k", mfc="none", ms=10, mew=2, label="Forward Projector"),
           plt.Line2D([], [], marker="x", ls="", color="k", ms=9, mew=2, label="hand calculation"),
           plt.Rectangle((0, 0), 1, 1, color="0.7", label="vertebra model, projected"),
           plt.Line2D([], [], marker="+", ls="", color="k", ms=10, mew=1.5, label="detector center")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(config.OUTPUT_DIR, "task4_markers_on_detectors.png"), dpi=150)
print("\nsaved: outputs/markers_projected_task4.csv, markers_projected_task4.npz, "
      "task4_marker_projection.txt, task4_markers_on_detectors.png")
