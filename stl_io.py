"""
stl_io.py - read the STL surface model of the vertebra with numpy only.

An STL file describes a surface as a list of triangles. Binary STL layout:
80-byte header, a 4-byte triangle count, then for every triangle 12 float32
numbers (a normal vector and the 3 corner points) and a 2-byte attribute.

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import struct
import numpy as np


def read_binary_stl(path):
    """Return the triangles as an array of shape (N, 3 corner points, 3 coordinates), in mm."""
    with open(path, "rb") as f:
        raw = f.read()
    n_triangles = struct.unpack("<I", raw[80:84])[0]
    if len(raw) != 84 + 50 * n_triangles:
        raise ValueError("not a binary STL (file size does not match the triangle count)")
    record = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attr", "<u2")])
    data = np.frombuffer(raw[84:], dtype=record, count=n_triangles)
    return data["corners"].astype(float)


def corner_points(triangles):
    """All distinct corner points of the triangles, shape (K, 3)."""
    return np.unique(triangles.reshape(-1, 3), axis=0)
