"""
projector.py - Task 3: Forward Projector (project a point from CK frame onto the Detectors).

X-ray imaging geometry (Discrete Tomography slides): the X-ray source S is a
point source; each photon beam is modelled as a straight line from S, through
the target object, to the X-ray detector plane. The image of a point is
    Td = Line(v, S) X Plane(Cd, d3)        (slide notation: line-plane intersection)
Math Primer Part 1, "Intersection of line and plane" (use case: X-ray imaging):
    line : L(t) = S + t (P - S)             t = 0 at the source, t = 1 at the point P
    plane: through the detector center C, with normal = the detector's w axis
    t = ((C - S) . w) / ((P - S) . w),     Td = S + t (P - S)
Finally Td is transformed into the detector frame, giving (u, v). Its w must be 0.

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import numpy as np
import config, frames
from geometry import line_plane_intersection, apply_transform


def project_to_detector(points_CK, pose):
    """
    Project one point (shape (3,)) or many points (shape (N, 3)), given in the
    CK frame in mm, onto the detector of pose "A" or "B".
    Returns the projected point(s) in the detector frame: (u, v) in mm.
    """
    det = frames.detector_geometry(pose)
    S = det["S_CK"]                                     # X-ray source
    P = np.asarray(points_CK, dtype=float)              # point(s) to project
    direction = P - S                                   # line from the source through the point
    # +w points from the detector to the source, so a line that travels from the
    # source toward the detector has direction . w < 0. Otherwise the point is
    # at or behind the source and its line never reaches the detector.
    if np.any(direction @ det["w"] >= 0):
        raise ValueError("point is at or behind the X-ray source: its line never reaches the detector")
    # Math Primer line-plane intersection with P = S, v = direction, A = C, n = w
    Td_CK, _ = line_plane_intersection(S, direction, det["C_CK"], det["w"])
    Td_Det = apply_transform(frames.F_Det_from_CK(pose), Td_CK)   # CK frame -> detector frame
    assert np.all(np.abs(Td_Det[..., 2]) < 1e-6), "projected point is not on the detector (w != 0)"
    return Td_Det[..., :2]


def detector_to_image(uv_mm):
    """(u, v) on the detector in mm -> (i1, i2) in the image frame, in pixels."""
    uv = np.asarray(uv_mm, dtype=float)
    uvw = np.concatenate([uv, np.zeros(uv.shape[:-1] + (1,))], axis=-1)   # add w = 0
    return apply_transform(frames.F_Img_from_Det(), uvw)[..., :2]


def project_to_image(points_CK, pose):
    """Project CK point(s) onto the detector, then give the result in image pixels (i1, i2)."""
    return detector_to_image(project_to_detector(points_CK, pose))


def on_detector(uv_mm):
    """True where the projected point falls inside the 200 x 200 mm detector."""
    half = config.DETECTOR_SIZE_mm / 2
    return np.all(np.abs(np.asarray(uv_mm)) <= half, axis=-1)
