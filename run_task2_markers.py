"""
Task 2 runner (Marker Transforms): transform the three markers through the
chain of relevant frames (MD -> CK -> Detector A and Detector B), save the
results for further use, and draw them so we can inspect them visually.

Test (assignment): "Inspect the results, visually, graphically."
The figure has two parts:
  (a) the whole setup seen from the feet of the patient, to compare with the
      CK frame figure in the assignment (sources, detectors, v and w axes);
  (b) the vertebra model and the three markers in the CK frame, seen from the
      feet of the patient, to compare with the vertebra-in-CK-frame figure in
      the assignment (M1 on the patient's right, M3 pointing to the table).

Run from the A1 folder (Mac Terminal):   python3 run_task2_markers.py

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
from markers import markers_in_all_frames, save_markers
from geometry import apply_transform
from stl_io import read_binary_stl, corner_points

# ---- transform and save ----
markers = markers_in_all_frames()          # dictionary: frame name -> (3 markers, 3 coordinates) in mm
save_markers(markers)                      # outputs/markers_task2.npz (for code) and .csv (for people)

lines = [f"MD->CK mode: {config.MD_TO_CK_MODE}", "",
         "marker | frame | coordinates (mm)"]
for i, name in enumerate(config.MARKER_NAMES):
    for frame_name in ("MD", "CK", "DetA", "DetB"):
        c = markers[frame_name][i]
        axes = "(u, v, w)" if frame_name.startswith("Det") else "(x, y, z)"
        lines.append(f"{name:6s} | {frame_name:5s} | {axes} = ({c[0]:9.3f}, {c[1]:9.3f}, {c[2]:9.3f})")
text = "\n".join(lines)
print(text)
with open(os.path.join(config.OUTPUT_DIR, "task2_marker_transforms.txt"), "w") as f:
    f.write(text + "\n")

# ---- picture ----
vertebra_MD = corner_points(read_binary_stl(config.STL_PATH))          # STL corner points, MD frame
vertebra_CK = apply_transform(frames.F_CK_from_MD(), vertebra_MD)      # same points, CK frame
marker_colors = {"M1": "tab:red", "M2": "tab:green", "M3": "tab:blue"}
pose_colors = {"A": "darkorange", "B": "purple"}

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5.6))

# (a) whole setup, seen from the feet of the patient: +z points into the page.
#     We plot CK x horizontally and CK y vertically, then flip the x axis so that
#     +x (the patient's right) is on our left, exactly as in the assignment figure.
for pose in ("A", "B"):
    det = frames.detector_geometry(pose)
    S, C, v, w = det["S_CK"], det["C_CK"], det["v"], det["w"]
    color = pose_colors[pose]
    ends = np.array([C - 100 * v, C + 100 * v])               # the 200 mm detector, seen edge-on
    ax_a.plot(ends[:, 0], ends[:, 1], color=color, lw=6, solid_capstyle="butt")
    for end in ends:                                          # X-ray beam: source to the two detector edges
        ax_a.plot([S[0], end[0]], [S[1], end[1]], color=color, lw=0.7, alpha=0.6)
    ax_a.plot([S[0], C[0]], [S[1], C[1]], "--", color=color, lw=1)      # beam central line
    ax_a.plot(S[0], S[1], "o", color=color, ms=9)
    ax_a.annotate(f"$S_{pose}$", S[:2], xytext=(0, 10), textcoords="offset points",
                  ha="center", fontsize=12, color=color)
    # detector axes v (red) and w (blue), 250 mm long, drawn at the detector center
    for axis, axis_name, axis_color in ((v, "v", "red"), (w, "w", "blue")):
        ax_a.annotate("", xy=C[:2] + 250 * axis[:2], xytext=C[:2],
                      arrowprops=dict(arrowstyle="-|>", color=axis_color, lw=1.8))
        label_at = C[:2] + 290 * axis[:2] - (70 * v[:2] if axis_name == "w" else 0)   # w label: moved off the beam central line
        ax_a.annotate(f"${axis_name}_{pose}$", label_at, color=axis_color,
                      fontsize=12, ha="center", va="center",
                      bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    ax_a.annotate(f"Detector {pose}", C[:2] - 230 * w[:2], color=color, fontsize=11,
                  ha="center", va="center")
# CK axes x and y, 300 mm long, from the CK center
for axis, label, label_position in (((1, 0), "x (right)", (300, 90)), ((0, 1), "y (anterior)", (0, 370))):
    tip = 300 * np.array(axis)
    ax_a.annotate("", xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.8))
    ax_a.annotate(label, label_position, fontsize=11, ha="center", va="center")
ax_a.set_xlim(1100, -1100)                 # flipped: +x on our left
ax_a.set_ylim(-1100, 1100)
ax_a.set_xticks(np.arange(-1000, 1001, 500)); ax_a.set_yticks(np.arange(-1000, 1001, 500))
ax_a.set_aspect("equal"); ax_a.grid(alpha=0.3)
ax_a.set_xlabel("CK x (mm), +x = patient's right"); ax_a.set_ylabel("CK y (mm), +y = anterior")
ax_a.set_title("(a) Setup seen from the feet of the patient", pad=10)

# (b) close-up of the vertebra model and the markers in the CK frame, same view
ax_b.scatter(vertebra_CK[:, 0], vertebra_CK[:, 1], s=0.05, c="0.6", rasterized=True)
for i, name in enumerate(config.MARKER_NAMES):
    M = markers["CK"][i]
    ax_b.plot(M[0], M[1], "o", color=marker_colors[name], ms=10, mfc="none", mew=2.2)
    ax_b.annotate(name, M[:2], xytext=(9, -14), textcoords="offset points",
                  color=marker_colors[name], fontsize=13, fontweight="bold")
ax_b.plot(0, 0, "k+", ms=14, mew=2)        # CK center
ax_b.annotate("CK center", (0, 0), xytext=(-8, 8), textcoords="offset points", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
ax_b.set_xlim(60, -60)                     # flipped: +x on our left, as in (a)
ax_b.set_ylim(-60, 50)
ax_b.set_xticks(np.arange(-60, 61, 20)); ax_b.set_yticks(np.arange(-60, 51, 10))
ax_b.set_aspect("equal"); ax_b.grid(alpha=0.3)
ax_b.set_xlabel("CK x (mm), +x = patient's right"); ax_b.set_ylabel("CK y (mm), +y = anterior")
ax_b.set_title("(b) Vertebra model and markers (close-up)", pad=10)

fig.tight_layout()
fig.savefig(os.path.join(config.OUTPUT_DIR, "task2_markers_in_CK.png"), dpi=150)
print("\nsaved: outputs/markers_task2.csv, markers_task2.npz, task2_marker_transforms.txt, task2_markers_in_CK.png")
