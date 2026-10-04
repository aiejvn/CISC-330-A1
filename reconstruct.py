"""
reconstruct.py - Task 7: Marker Reconstruction.

"Develop a module to reconstruct a point in CK frame from a pair of points
given in their respective Detector frames. Also compute appropriate residual
error metric (REM)."

Steps (Discrete Tomography slide "Reconstruction of point marker from 2 images"):
  1. transform each point from its detector frame to the CK (home) frame;
  2. the X-ray sources S_A and S_B are already known in the CK frame;
  3. write up the back-projection lines L_A and L_B
     (slide "Back-projection":  L = v*t + Td,  v = (S - Td) / |S - Td|);
  4. compute the symbolic intersection of the two lines -> reconstructed point M;
  5. REM = shortest perpendicular distance between the two back-projection
     lines (assignment p.7 "as we did in class"; Math Primer Part 1 slide:
     "Residual Error Metric (REM): distance between the two lines").

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import numpy as np
import frames
from geometry import apply_transform, normalize, symbolic_intersection


def back_projection_line(uv_mm, pose):
    """
    Back-projection line of a point (u, v) on the detector of pose "A" or "B"
    (Discrete Tomography slide "Back-projection": L = v*t + Td, v = (S - Td)/|S - Td|).
    Returns (Td_CK, direction_CK): the point on the detector and the unit
    direction toward the X-ray source, both in the CK frame.
    (We call the slide's v "direction" in the code so that it is not mixed up
    with the detector's v axis.)
    """
    uv = np.asarray(uv_mm, dtype=float)
    Td_Det = np.array([uv[0], uv[1], 0.0])                        # on the detector: w = 0
    Td_CK = apply_transform(frames.F_CK_from_Det(pose), Td_Det)   # detector frame -> CK frame
    S_CK = frames.detector_geometry(pose)["S_CK"]                 # X-ray source of this pose
    return Td_CK, normalize(S_CK - Td_CK)


def reconstruct_point(uv_A_mm, uv_B_mm):
    """
    Reconstruct one point in the CK frame from its location on detector A and
    on detector B (each (u, v) in mm, in its own detector frame).
    Line 1 comes from detector A and line 2 from detector B (Math Primer names:
    L1 = P1 + t1 v1, L2 = P2 + t2 v2).
    Returns (M_CK, REM_mm, lines), where `lines` keeps both back-projection
    lines and their closest points L1, L2 (used for the picture).
    """
    P1, v1 = back_projection_line(uv_A_mm, "A")
    P2, v2 = back_projection_line(uv_B_mm, "B")
    M, REM, L1, L2 = symbolic_intersection(P1, v1, P2, v2)   # REM = d = |L1 - L2|
    lines = {"A": (P1, v1, L1), "B": (P2, v2, L2)}
    return M, REM, lines


def reconstruct_markers(uv_A_list_mm, uv_B_list_mm):
    """Reconstruct several markers; the i-th point on detector A is paired with the i-th point on detector B."""
    results = [reconstruct_point(uv_A, uv_B) for uv_A, uv_B in zip(uv_A_list_mm, uv_B_list_mm)]
    M_CK = np.array([result[0] for result in results])
    REM_mm = np.array([result[1] for result in results])
    return M_CK, REM_mm
