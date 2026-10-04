"""
markers.py - Task 2: carry the three markers through the chain of frames.

A frame transformation is the tool (a 4x4 matrix); here we APPLY it to the
three marker centers and save the results, because later tasks (projection,
DRR, reconstruction, error analysis) all need the true marker positions.

Note: a marker's coordinates in a detector frame describe the marker itself in
that frame (its w coordinate is its distance from the detector plane, about
1000 mm). This is NOT where the marker is projected onto the detector; the
projection is computed by the Forward Projector (Tasks 3-4).

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import os
import numpy as np
import config, frames
from geometry import apply_transform


def markers_in_all_frames(mode=None):
    """Marker centers in the MD, CK, DetA and DetB frames (mm), each shape (3, 3)."""
    M_MD = config.MARKERS_MD_mm
    M_CK = apply_transform(frames.F_CK_from_MD(mode), M_MD)
    return {"MD": M_MD,
            "CK": M_CK,
            "DetA": apply_transform(frames.F_Det_from_CK("A"), M_CK),
            "DetB": apply_transform(frames.F_Det_from_CK("B"), M_CK)}


def save_markers(markers, basename="markers_task2"):
    """Save the markers as .npz (read by later tasks) and .csv (easy to open and read)."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.savez(os.path.join(config.OUTPUT_DIR, basename + ".npz"), **markers)
    rows = ["marker,frame,c1_mm,c2_mm,c3_mm"]
    for frame_name, coords in markers.items():
        for name, c in zip(config.MARKER_NAMES, coords):
            rows.append(f"{name},{frame_name},{c[0]:.4f},{c[1]:.4f},{c[2]:.4f}")
    with open(os.path.join(config.OUTPUT_DIR, basename + ".csv"), "w") as f:
        f.write("\n".join(rows) + "\n")


def load_markers(basename="markers_task2"):
    """Load the markers saved by Task 2 (dictionary: frame name -> (3, 3) array)."""
    data = np.load(os.path.join(config.OUTPUT_DIR, basename + ".npz"))
    return {k: data[k] for k in data.files}
