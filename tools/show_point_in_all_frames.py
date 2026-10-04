"""
show_point_in_all_frames.py - print any point's coordinates in every frame.

HOW TO USE: edit the list POINTS below (label, the frame you are giving the
point in: "MD" or "CK", and its x, y, z in mm), save, then run:
    python3 tools/show_point_in_all_frames.py

For each point it prints:
  - the point itself in the MD, CK, DetA and DetB frames (frame transformations, Task 1-2);
  - its projection onto detector A and detector B, in the detector frame (u, v mm)
    and in the image frame (i1, i2 pixels) (forward projector, Task 3-4).

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, frames
from geometry import apply_transform
from projector import project_to_detector, detector_to_image, on_detector

# ------------------------- edit this list -------------------------
POINTS = [
    ("CK center",  "CK", [0, 0, 0]),
    ("M1",         "MD", [-19, 28, 11]),
    ("M2",         "MD", [43, 14, 2]),
    ("M3",         "MD", [2, 48, -6]),
    ("10 mm up z", "CK", [0, 0, 10]),
]
# -------------------------------------------------------------------

np.set_printoptions(precision=3, suppress=True)
print(f"MD->CK mode: {config.MD_TO_CK_MODE}   (units: mm, image frame in pixels)\n")
for label, frame, xyz in POINTS:
    xyz = np.asarray(xyz, dtype=float)
    P_CK = apply_transform(frames.F_CK_from_MD(), xyz) if frame == "MD" else xyz
    P_MD = apply_transform(frames.F_MD_from_CK(), P_CK)
    print(f"== {label} (given in {frame})")
    print(f"   MD   (x, y, z) = {P_MD}")
    print(f"   CK   (x, y, z) = {P_CK}")
    for p in ("A", "B"):
        P_Det = apply_transform(frames.F_Det_from_CK(p), P_CK)
        uv = project_to_detector(P_CK, p)
        px = detector_to_image(uv)
        print(f"   Det{p} (u, v, w) = {P_Det}    projection onto detector {p}: (u, v) = {uv} mm"
              f" = (i1, i2) {px} px{'' if on_detector(uv) else '   (outside the detector!)'}")
    print()
