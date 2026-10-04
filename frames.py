"""
frames.py - Task 1: frame transformations between MD, CK, detector and image frames.

Every transformation is a 4x4 homogeneous matrix named F_<to>_from_<from>,
which is the course's F_{to <- from} (Math Primer Part 2: capital F with the
"to" frame first and the "from" frame second, read right to left).
Matrices are chained right to left (the right-most one acts first), e.g. the
coordinates of an MD point in the frame of detector A:
    p_DetA = F_DetA_from_CK @ F_CK_from_MD @ p_MD
This only CHANGES COORDINATES of a 3D point (its w is about 1000 mm, its
distance from the detector plane). It does NOT tell where the point lands on
the detector. For that use projector.project_to_detector (the line from the
X-ray source through the point, intersected with the detector plane), and only
then F_Img_from_Det to go from detector mm to image pixels.

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import numpy as np
import config
from geometry import (rotation_z, translation, scaling, frame_to_home,
                      invert_rigid, apply_transform)


# ---- MD <-> CK ------------------------------------------------------------------
def F_CK_from_MD(mode=None):
    """
    MD frame -> CK frame. The centers of the two frames coincide (assignment
    p.4), so there is no translation.
      mode "identity" : the base vectors are identical.
      mode "rotate180": rotation of 180 deg about +z, (x, y, z)_CK = (-x, -y, z)_MD
                        (see config.py for why this is our main setting).
    """
    mode = config.MD_TO_CK_MODE if mode is None else mode
    if mode == "identity":
        return np.eye(4)
    if mode == "rotate180":
        return rotation_z(180.0)
    raise ValueError(f"unknown MD_TO_CK_MODE: {mode}")


def F_MD_from_CK(mode=None):
    """CK frame -> MD frame: the inverse of F_CK_from_MD."""
    return invert_rigid(F_CK_from_MD(mode))


# ---- X-ray source and detector in the CK frame -------------------------------------------
def detector_geometry(pose):
    """
    X-ray source position, detector center and detector axes (u, v, w), all in
    CK coordinates, for pose "A" or "B".

    Start from the hypothetical zero-angle home position (assignment p.2):
      source S0 on +y, SAD = 1000 mm from the CK center (ceiling side),
      detector center C0 on -y, SDD - SAD = 1000 mm from the CK center,
      +u = CK +z, +v = CK +x, +w = CK +y (+w points to the source).
    Then rotate all of it about +z by the pose angle (+45 for A, -45 for B).
    Returned keys (names as in the assignment figure: S_A, C_A, u, v, w):
      "S_CK": X-ray source,  "C_CK": detector center,  "u", "v", "w": unit axes.
    """
    Rz = rotation_z(config.POSE_ANGLE_deg[pose])[:3, :3]          # 3x3 rotation about +z
    S = Rz @ np.array([0.0, config.SAD_mm, 0.0])                   # source at the home position, rotated
    C = Rz @ np.array([0.0, -(config.SDD_mm - config.SAD_mm), 0.0])  # detector center, rotated
    u = np.array([0.0, 0.0, 1.0])                                  # +u is always CK +z
    v = Rz @ np.array([1.0, 0.0, 0.0])                             # +v is CK +x at the home position
    w = Rz @ np.array([0.0, 1.0, 0.0])                             # +w is CK +y at the home position
    return {"S_CK": S, "C_CK": C, "u": u, "v": v, "w": w}


# ---- CK <-> detector ----------------------------------------------------------------
def F_CK_from_Det(pose):
    """Detector frame -> CK frame: axes u, v, w as columns, detector center C in the last column."""
    det = detector_geometry(pose)
    return frame_to_home(det["C_CK"], det["u"], det["v"], det["w"])


def F_Det_from_CK(pose):
    """CK frame -> detector frame: the inverse (shift by -C, then axes u, v, w as rows)."""
    return invert_rigid(F_CK_from_Det(pose))


# ---- detector <-> image ----------------------------------------------------------------
def F_Img_from_Det():
    """
    Detector frame (mm, center at the detector center) -> image frame (pixels,
    center at the lower-left corner of the image). Same for both detectors.
    Discrete Tomography slides: Step 1 scale (mm -> pixels), Step 2 shift:
        i1 = u / 0.1 + 1000,  i2 = v / 0.1 + 1000,  i3 = w / 0.1
    Pixel (column k, row j) covers i1 in [k, k+1) and i2 in [j, j+1), so its
    middle is at (k + 0.5, j + 0.5); k and j run from 0 to 1999 (i1 = 2000 is
    the right edge of the image, not a pixel).
    """
    scale = scaling(1.0 / config.PIXEL_SIZE_mm)
    half = config.N_PIXELS / 2.0
    shift = translation([half, half, 0.0])
    return shift @ scale


def F_Det_from_Img():
    """Image frame -> detector frame: undo the shift, then undo the scale."""
    half = config.N_PIXELS / 2.0
    return scaling(config.PIXEL_SIZE_mm) @ translation([-half, -half, 0.0])


# ---- convenience ------------------------------------------------------------------------
def all_transforms(mode=None):
    """Dictionary of every frame transformation needed in HW1, keyed 'to<-from'."""
    F = {"CK<-MD": F_CK_from_MD(mode), "MD<-CK": F_MD_from_CK(mode)}
    for pose in ("A", "B"):
        F[f"CK<-Det{pose}"] = F_CK_from_Det(pose)
        F[f"Det{pose}<-CK"] = F_Det_from_CK(pose)
        F[f"Img{pose}<-Det{pose}"] = F_Img_from_Det()
        F[f"Det{pose}<-Img{pose}"] = F_Det_from_Img()
    return F


def transform_points(points, chain):
    """Apply a list of 4x4 transformations in order (the first one in the list acts first)."""
    result = np.asarray(points, float)
    for F in chain:
        result = apply_transform(F, result)
    return result
