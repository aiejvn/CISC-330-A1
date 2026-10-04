"""
check_stl_markers.py - measure the three marker balls that are inside the STL file.

Why
---
The assignment (p.3) says the markers are three identical spherical steel
markers, each 3 mm in diameter, at the positions in the marker table (Model
frame). The STL file "LumbarVertebrae.-witrh-Markers.stl" contains three small
balls. This script measures them straight from the file, so anyone can re-run
it and see the same numbers.

Steps
-----
1. Read all triangles of the STL surface model.
2. Split the surface into separate pieces. Two corner points are in the same
   piece if a triangle edge connects them. We "flood fill" from one corner
   point to every corner point reachable along triangle edges (like filling a
   connected region in an image). The vertebra is one big piece; each marker
   ball is its own small, closed piece.
3. For each ball-sized piece:
   - its width in x, y and z (for a sphere, all three equal the diameter);
   - its center and radius with the Math Primer Part 1 slide "Average fitting
     of sphere to points set":  C = avg(Pi),  R = avg(|Pi - C|);
   - the smallest and largest |Pi - C|: equal values mean a perfect sphere.
4. Compare with the assignment: 3 mm diameter, centers at the table values.
5. Show what the difference means on the detectors (project both centers).
6. Save a picture of each ball with the assignment's 3 mm sphere drawn on top.

Run from the A1 folder (Mac Terminal):   python3 tools/check_stl_markers.py

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import os, sys
from collections import deque
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config, frames
from stl_io import read_binary_stl
from geometry import apply_transform
from projector import project_to_detector

lines = []
def say(text=""):
    print(text); lines.append(text)

# ---- step 1: read the STL -------------------------------------------------------------
triangles = read_binary_stl(config.STL_PATH)                 # shape (N, 3 corners, xyz)
say(f"STL file: {os.path.basename(config.STL_PATH)}, {len(triangles)} triangles (Model frame, mm)")

# ---- step 2: split into separate pieces (flood fill along triangle edges) -------------
# Number every distinct corner point; each triangle then lists 3 corner numbers.
corners, corner_number = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
corner_number = corner_number.reshape(-1, 3)
neighbours = [[] for _ in range(len(corners))]              # which corners share an edge with each corner
for a, b, c in corner_number:
    neighbours[a] += [b, c]; neighbours[b] += [a, c]; neighbours[c] += [a, b]

piece_of_corner = np.full(len(corners), -1)                  # -1 = not visited yet
n_pieces = 0
for start in range(len(corners)):
    if piece_of_corner[start] != -1:
        continue                                             # already belongs to a piece
    piece_of_corner[start] = n_pieces
    to_visit = deque([start])
    while to_visit:                                          # visit everything reachable from `start`
        k = to_visit.popleft()
        for m in neighbours[k]:
            if piece_of_corner[m] == -1:
                piece_of_corner[m] = n_pieces
                to_visit.append(m)
    n_pieces += 1
corner_count = np.bincount(piece_of_corner)
say(f"the surface model has {n_pieces} separate pieces with {sorted(corner_count.tolist(), reverse=True)} corner points")

# ---- step 3: measure the ball-sized pieces ---------------------------------------------------
balls = []
for piece in range(n_pieces):
    P = corners[piece_of_corner == piece]                    # corner points of this piece
    width = P.max(axis=0) - P.min(axis=0)                    # width in x, y, z
    if width.max() > 10 or width.max() < 1:                  # skip the vertebra and tiny leftover bits
        continue
    C = P.mean(axis=0)                                       # Math Primer: C = avg(Pi)
    dist = np.linalg.norm(P - C, axis=1)                     # |Pi - C|
    R = dist.mean()                                          # Math Primer: R = avg(|Pi - C|)
    if dist.max() - dist.min() > 0.05 * R:
        say(f"  (skipping a piece that is not a sphere: {len(P)} corner points near {np.round(C, 1)})")
        continue
    balls.append({"C": C, "R": R, "width": width, "dmin": dist.min(), "dmax": dist.max(), "P": P})

say("\nmarker balls in the STL vs the assignment's marker table (Model frame, mm):")
say("marker | table center     | STL ball center (avg of corners) | distance | STL diameter 2R | smallest/largest |Pi-C| | width in x, y, z")
matched = []
for name, table in zip(config.MARKER_NAMES, config.MARKERS_MD_mm):
    ball = min(balls, key=lambda b: np.linalg.norm(b["C"] - table))      # the ball closest to this table entry
    matched.append(ball)
    C = ball["C"]
    say(f"{name:6s} | {str(table):16s} | ({C[0]:8.3f}, {C[1]:7.3f}, {C[2]:7.3f})       | {np.linalg.norm(C - table):.3f}    | "
        f"{2 * ball['R']:.4f}          | {ball['dmin']:.4f} / {ball['dmax']:.4f}        | {np.round(ball['width'], 3)}")
say("assignment: 'three identical spherical markers (M1, M2, M3), each 3 mm in diameter' (p.3)")

# ---- step 5: what this difference means on the detectors -----------------------------------
say("\nprojected on the detectors (forward projector, MD->CK mode = " + config.MD_TO_CK_MODE + "):")
say("marker det | table center -> (u, v) mm | STL ball center -> (u, v) mm | difference mm | difference px")
F = frames.F_CK_from_MD()
for name, table, ball in zip(config.MARKER_NAMES, config.MARKERS_MD_mm, matched):
    for p in ("A", "B"):
        uv_table = project_to_detector(apply_transform(F, table), p)
        uv_stl = project_to_detector(apply_transform(F, ball["C"]), p)
        d = np.linalg.norm(uv_stl - uv_table)
        say(f"{name:6s} {p:3s} | ({uv_table[0]:7.3f}, {uv_table[1]:7.3f})       | ({uv_stl[0]:7.3f}, {uv_stl[1]:7.3f})          | "
            f"{d:.3f}         | {d / config.PIXEL_SIZE_mm:.1f}")

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
with open(os.path.join(config.OUTPUT_DIR, "stl_marker_check.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

# ---- step 6: picture --------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
angle = np.linspace(0, 2 * np.pi, 200)
for ax, name, table, ball in zip(axs, config.MARKER_NAMES, config.MARKERS_MD_mm, matched):
    C, R, P = ball["C"], ball["R"], ball["P"]
    ax.scatter(P[:, 0], P[:, 1], s=3, c="0.3", label="corner points of the STL ball (x-y view)")
    ax.plot(table[0] + 1.5 * np.cos(angle), table[1] + 1.5 * np.sin(angle), "r--",
            label="assignment: 3 mm sphere at the table position")
    ax.plot(C[0] + R * np.cos(angle), C[1] + R * np.sin(angle), "b-", lw=1,
            label=f"STL ball: C = avg(Pi), 2R = {2 * R:.2f} mm")
    ax.plot(*table[:2], "r+", ms=12); ax.plot(*C[:2], "b+", ms=12)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"{name}: table ({table[0]:.0f}, {table[1]:.0f}, {table[2]:.0f})\n"
                 f"STL ball center ({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f})")
    ax.set_xlabel("MD x (mm)"); ax.set_ylabel("MD y (mm)")
axs[0].legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(config.OUTPUT_DIR, "stl_marker_balls_vs_assignment.png"), dpi=110)
print("\nsaved: outputs/stl_marker_check.txt, outputs/stl_marker_balls_vs_assignment.png")
