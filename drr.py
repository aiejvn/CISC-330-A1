"""
drr.py - Task 5: Digitally Reconstructed Radiograph (sphere phase, then vertebra phase).

For every pixel of a detector, one ray goes from the X-ray source S through the
middle of the pixel. The pixel value is the slides' sum
    ln(I0 / I_n) / k = d_1 x_1 + d_2 x_2 + ... + d_n x_n = sum_i d_i x_i
with d_i the length of the ray inside object i (in cm) and x_i its density (g/cm^3).
We store this sum (g/cm^2); I0, I and k are not needed to build the DRR.

Objects (sphere phase): a sphere (R = 50 mm, density 0.25) at the CK center and
three steel balls (2 mm diameter) at the marker positions, all inside the sphere.
Along a ray, a ball REPLACES sphere material, so
    sum d x = x_sphere * (l_sphere - sum l_ball) + x_steel * sum l_ball.

Array convention: img[j, k] with k = image column = i1 (detector u direction) and
j = image row = i2 (detector v direction). Plot with origin="lower".

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import numpy as np
import config, frames
from geometry import apply_transform, line_sphere_chord

MARKER_DIAMETER_mm = 2.0   # all markers are 2 mm (the STL balls too)


def n_pixels(pixel_mm):
    """Pixels per side for a given pixel size (0.1 mm -> 2000; coarse 1 mm -> 200)."""
    return int(round(config.DETECTOR_SIZE_mm / pixel_mm))


def pixel_centers_CK(pose, pixel_mm, rows):
    """
    Middle of every pixel in the given image rows, in the CK frame, shape (len(rows), n, 3).
    Pixel (k, j) covers [k, k+1) x [j, j+1) pixels, so its middle is at (k+0.5, j+0.5);
    in detector mm: u = (k+0.5)*pixel_mm - 100, v = (j+0.5)*pixel_mm - 100, w = 0.
    """
    n = n_pixels(pixel_mm)
    half = config.DETECTOR_SIZE_mm / 2
    u = (np.arange(n) + 0.5) * pixel_mm - half
    v = (np.asarray(rows) + 0.5) * pixel_mm - half
    U, V = np.meshgrid(u, v)                                   # shape (len(rows), n): [row j, column k]
    Td_Det = np.stack([U, V, np.zeros_like(U)], axis=-1).reshape(-1, 3)
    return apply_transform(frames.F_CK_from_Det(pose), Td_Det).reshape(len(rows), n, 3)


def check_markers_inside_sphere(marker_centers_CK):
    """The 'replacement' formula needs every ball completely inside the sphere."""
    reach = np.linalg.norm(marker_centers_CK, axis=1) + MARKER_DIAMETER_mm / 2
    if np.any(reach >= config.SPHERE_RADIUS_mm):
        raise ValueError(f"a marker ball sticks out of the sphere (reach {reach} mm)")


def render_drr(pose, pixel_mm, marker_centers_CK, with_markers=True, chunk_rows=50):
    """
    DRR of the sphere (+ markers) for pose "A" or "B": sum d_i x_i in g/cm^2, shape (n, n), indexed [j, k].
    Processed in chunks of rows so that the full 2000 x 2000 image fits in memory.
    """
    marker_centers_CK = np.asarray(marker_centers_CK, float)
    check_markers_inside_sphere(marker_centers_CK)
    S = frames.detector_geometry(pose)["S_CK"]
    n = n_pixels(pixel_mm)
    r_ball = MARKER_DIAMETER_mm / 2
    img = np.zeros((n, n))
    for r0 in range(0, n, chunk_rows):
        rows = np.arange(r0, min(n, r0 + chunk_rows))
        P = pixel_centers_CK(pose, pixel_mm, rows).reshape(-1, 3)
        d = P - S
        d /= np.linalg.norm(d, axis=1, keepdims=True)                          # unit direction S -> pixel
        l_sphere = line_sphere_chord(S, d, np.zeros(3), config.SPHERE_RADIUS_mm)   # mm
        l_balls = np.zeros_like(l_sphere)
        if with_markers:
            for M in marker_centers_CK:
                l_balls += line_sphere_chord(S, d, M, r_ball)                  # balls are disjoint
        sum_dx = (config.SPHERE_DENSITY_g_cm3 * (l_sphere - l_balls)
                  + config.STEEL_DENSITY_g_cm3 * l_balls) / 10.0               # mm -> cm
        img[rows] = sum_dx.reshape(len(rows), n)
    return img


# ---------------------------------------------------------------------------------
# Vertebra phase: rays against a triangle surface model (numpy only)
# ---------------------------------------------------------------------------------
def mesh_path_length(triangles_CK, pose, pixel_mm):
    """
    Length (mm) of every pixel ray inside the closed surface made of `triangles_CK`
    (shape (N, 3 corners, 3), CK frame, normals pointing outward by the
    counter-clockwise rule; a cavity has inward normals and subtracts itself).
    Returns (length, balance), both shape (n, n) indexed [j, k]. `balance` counts
    entries minus exits per ray and must be 0 for a closed surface.

    Moeller-Trumbore for the ray S + t d (|d| = 1) and a triangle (v0, v1, v2):
        e1 = v1 - v0, e2 = v2 - v0, s = S - v0
        a = e1 . (d x e2) = d . (e2 x e1)
        u = (s . (d x e2)) / a = d . (e2 x s) / a
        v = d . (s x e1) / a
        t = e2 . (s x e1) / a
    hit if u >= 0, v >= 0, u + v <= 1, t > 0.
    Length inside = sum over hits of sign(n . d) * t (an exit face adds t, an entry
    face subtracts t); the outward normal n ~ e1 x e2 gives n . d = -a, so the
    sign is sign(-a). No sorting of the hits is needed.
    Speed: every triangle is projected onto the detector and only the pixels in
    the bounding box of its projection are tested (the others cannot be hit).
    The terms d . g with g = e2 x e1, e2 x s, s x e1 are constant per triangle.
    """
    from projector import project_to_detector              # (kept local: projector imports config, frames only)
    S = frames.detector_geometry(pose)["S_CK"]
    n = n_pixels(pixel_mm)
    D = pixel_centers_CK(pose, pixel_mm, np.arange(n)) - S    # (n, n, 3): S -> pixel middle
    D /= np.linalg.norm(D, axis=-1, keepdims=True)

    tri = np.asarray(triangles_CK, float)
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    s = S - v0
    G = np.stack([np.cross(e2, e1), np.cross(e2, s), np.cross(s, e1)], axis=2)   # (N, 3, 3): columns g_a, g_u, g_v/q
    t_num = np.einsum("ij,ij->i", e2, G[:, :, 2])                                # e2 . (s x e1)

    half = config.DETECTOR_SIZE_mm / 2
    pix = (project_to_detector(tri.reshape(-1, 3), pose) + half) / pixel_mm      # projected corners in pixel units
    pix = pix.reshape(-1, 3, 2)
    k0 = np.clip(np.floor(pix[:, :, 0].min(axis=1)).astype(int) - 1, 0, n - 1)
    k1 = np.clip(np.floor(pix[:, :, 0].max(axis=1)).astype(int) + 1, 0, n - 1)
    j0 = np.clip(np.floor(pix[:, :, 1].min(axis=1)).astype(int) - 1, 0, n - 1)
    j1 = np.clip(np.floor(pix[:, :, 1].max(axis=1)).astype(int) + 1, 0, n - 1)
    outside = ((pix[:, :, 0].max(axis=1) < 0) | (pix[:, :, 0].min(axis=1) > n)
               | (pix[:, :, 1].max(axis=1) < 0) | (pix[:, :, 1].min(axis=1) > n))

    length = np.zeros((n, n))
    balance = np.zeros((n, n), dtype=int)
    for i in np.flatnonzero(~outside):
        d = D[j0[i]:j1[i] + 1, k0[i]:k1[i] + 1]
        A, B, C = np.moveaxis(d @ G[i], -1, 0)                  # a, d.(e2 x s), d.(s x e1)
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v, t = B / A, C / A, t_num[i] / A
        hit = (np.abs(A) > 1e-14) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        sign = np.where(hit, -np.sign(A), 0.0)                  # +1 exit, -1 entry
        length[j0[i]:j1[i] + 1, k0[i]:k1[i] + 1] += sign * np.where(hit, t, 0.0)
        balance[j0[i]:j1[i] + 1, k0[i]:k1[i] + 1] += sign.astype(int)
    return length, balance


def render_vertebra_drr(pose, pixel_mm, bone_triangles_CK, ball_triangles_CK_list):
    """
    DRR (sum d_i x_i in g/cm^2) of the vertebra (density BONE_DENSITY) with the STL marker balls
    (steel density). Slides' formula sum d_i x_i, objects counted separately: the
    balls only touch the bone surface, so the overlap is a sliver and is ignored.
    Returns (DRR, info) where info holds the path lengths and the entry/exit balance.
    """
    l_bone, bal_bone = mesh_path_length(bone_triangles_CK, pose, pixel_mm)
    l_balls = np.zeros_like(l_bone)
    bal_balls = np.zeros_like(bal_bone)
    for tri in ball_triangles_CK_list:
        l, b = mesh_path_length(tri, pose, pixel_mm)
        l_balls += l; bal_balls += b
    sum_dx = (config.BONE_DENSITY_g_cm3 * l_bone + config.STEEL_DENSITY_g_cm3 * l_balls) / 10.0
    return sum_dx, {"l_bone": l_bone, "l_balls": l_balls, "bal_bone": bal_bone, "bal_balls": bal_balls}
