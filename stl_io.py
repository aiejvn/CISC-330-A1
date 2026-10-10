"""
stl_io.py - read the STL surface model of the vertebra with numpy only.

An STL file describes a surface as a list of triangles. Binary STL layout:
80-byte header, a 4-byte triangle count, then for every triangle 12 float32
numbers (a normal vector and the 3 corner points) and a 2-byte attribute.

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import struct
from collections import deque
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


def triangle_pieces(triangles):
    """
    Split the surface into separate pieces. Two triangles are in the same piece if
    their corner points are connected along triangle edges (flood fill, as in
    tools/check_stl_markers.py). Returns one piece number per triangle, shape (N,).
    """
    corners, number = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    number = number.reshape(-1, 3)
    neighbours = [[] for _ in range(len(corners))]
    for a, b, c in number:
        neighbours[a] += [b, c]; neighbours[b] += [a, c]; neighbours[c] += [a, b]
    piece_of_corner = np.full(len(corners), -1)
    n_pieces = 0
    for start in range(len(corners)):
        if piece_of_corner[start] != -1:
            continue
        piece_of_corner[start] = n_pieces
        to_visit = deque([start])
        while to_visit:
            k = to_visit.popleft()
            for m in neighbours[k]:
                if piece_of_corner[m] == -1:
                    piece_of_corner[m] = n_pieces
                    to_visit.append(m)
        n_pieces += 1
    return piece_of_corner[number[:, 0]]


def signed_volume(triangles):
    """
    Volume enclosed by a closed triangle surface: sum of v0 . (v1 x v2) / 6.
    Positive when the triangle normals (counter-clockwise rule) point OUTWARD,
    negative for a cavity (normals point inward).
    """
    return np.sum(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))) / 6.0


def split_vertebra_and_markers(triangles, marker_table_MD):
    """
    Separate the STL into the marker balls and everything else (the "bone group").
    A ball = a piece that is at most 3 mm wide with a clearly positive volume
    (about 4.19 mm^3 for a 2 mm ball). The other small pieces in the file (an
    internal cavity with negative volume, and a flat sliver) are kept in the bone
    group; with outward/inward normals the signed length rule handles them.
    Returns (bone_triangles, ball_triangles_list, ball_centers_MD), the balls in
    the order of the marker table (each table entry gets the closest ball);
    centre = average of the ball's corner points (Math Primer "average fitting").
    """
    piece = triangle_pieces(triangles)
    ball_ids, centers = [], []
    for p in np.unique(piece):
        tri = triangles[piece == p]
        P = tri.reshape(-1, 3)
        if np.ptp(P, axis=0).max() <= 3.0 and signed_volume(tri) > 1.0:
            ball_ids.append(p)
            centers.append(np.unique(P, axis=0).mean(axis=0))
    centers = np.array(centers)
    order = [int(np.argmin(np.linalg.norm(centers - m, axis=1))) for m in np.asarray(marker_table_MD)]
    if sorted(order) != list(range(len(centers))):
        raise ValueError("could not match the STL balls to the marker table one-to-one")
    ball_ids = [ball_ids[i] for i in order]
    is_ball = np.isin(piece, ball_ids)
    return triangles[~is_ball], [triangles[piece == p] for p in ball_ids], centers[order]


def enlarge_ball(ball_triangles, center, diameter_mm):
    """
    Scale a ball's surface about its own center so that its diameter becomes `diameter_mm`.
    The STL balls are 2 mm across (tools/check_stl_markers.py); the assignment says 3 mm.
    The diameter is measured as 2 x the mean distance of the corner points to the center.
    """
    P = ball_triangles.reshape(-1, 3)
    radius = np.linalg.norm(np.unique(P, axis=0) - center, axis=1).mean()
    return center + (ball_triangles - center) * (diameter_mm / 2.0 / radius)


def corner_points(triangles):
    """All distinct corner points of the triangles, shape (K, 3)."""
    return np.unique(triangles.reshape(-1, 3), axis=0)
