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
    u = (np.arange(n) + 0.5) * pixel_mm - half                 # u of every column: 0.5, 1.5, 2.5, ... pixels, in mm from the detector center
    v = (np.asarray(rows) + 0.5) * pixel_mm - half             # v of every requested row, the same way
    # meshgrid builds two tables: U repeats the u values on every row, V repeats each row's v across the columns,
    # so (U[j, k], V[j, k]) is the middle of pixel (k, j)
    U, V = np.meshgrid(u, v)                                   # shape (len(rows), n): [row j, column k]
    # glue U, V and w = 0 together as three coordinates, then flatten the grid into one long list of points
    # (reshape(-1, 3): "as many rows as needed, 3 values each")
    Td_Det = np.stack([U, V, np.zeros_like(U)], axis=-1).reshape(-1, 3)
    # move every point from detector coordinates to CK coordinates, then fold the list back into a grid
    return apply_transform(frames.F_CK_from_Det(pose), Td_Det).reshape(len(rows), n, 3)


def check_markers_inside_sphere(marker_centers_CK):
    """The 'replacement' formula needs every ball completely inside the sphere."""
    # norm(..., axis=1) = the length of every marker's position vector, i.e. its distance from the sphere center;
    # adding the ball radius gives how far the ball's outer side reaches
    reach = np.linalg.norm(marker_centers_CK, axis=1) + MARKER_DIAMETER_mm / 2
    if np.any(reach >= config.SPHERE_RADIUS_mm):                # np.any: True if at least one ball reaches out
        raise ValueError(f"a marker ball sticks out of the sphere (reach {reach} mm)")


def render_sphere_drr(pose, pixel_mm, marker_centers_CK, with_markers=True, chunk_rows=50):
    """
    DRR of the sphere (+ markers) for pose "A" or "B": sum d_i x_i in g/cm^2, shape (n, n), indexed [j, k].
    Processed in chunks of rows so that the full 2000 x 2000 image fits in memory.
    """
    marker_centers_CK = np.asarray(marker_centers_CK, float)
    check_markers_inside_sphere(marker_centers_CK)
    S = frames.detector_geometry(pose)["S_CK"]          # this pose's own source: nothing is shared between A and B
    n = n_pixels(pixel_mm)
    r_ball = MARKER_DIAMETER_mm / 2
    img = np.zeros((n, n))
    for r0 in range(0, n, chunk_rows):
        rows = np.arange(r0, min(n, r0 + chunk_rows))           # row numbers of this chunk (the last chunk may be shorter)
        P = pixel_centers_CK(pose, pixel_mm, rows).reshape(-1, 3)    # one target point per pixel (grid flattened to a list)
        d = P - S                                               # S is one point, P a list: S is subtracted from every point
        # divide every direction by its own length so that it has length 1 (keepdims keeps the lengths as a
        # column so that the division goes row by row)
        d /= np.linalg.norm(d, axis=1, keepdims=True)                          # unit direction S -> pixel
        # the chord (length of the ray inside a sphere) is the d_i of the slides' sum; we need it for the
        # sphere and for every ball, and it has an exact closed form (geometry.line_sphere_chord)
        l_sphere = line_sphere_chord(S, d, np.zeros(3), config.SPHERE_RADIUS_mm)   # mm
        l_balls = np.zeros_like(l_sphere)                      # zeros, same size as l_sphere: total ball path per ray
        if with_markers:
            for M in marker_centers_CK:
                l_balls += line_sphere_chord(S, d, M, r_ball)                  # balls are disjoint
        # inside a ball the ray sees steel instead of sphere material, so the ball chord is taken out of
        # the sphere chord (density 0.25) and counted with the steel density instead
        sum_dx = (config.SPHERE_DENSITY_g_cm3 * (l_sphere - l_balls)
                  + config.STEEL_DENSITY_g_cm3 * l_balls) / 10.0               # mm -> cm
        img[rows] = sum_dx.reshape(len(rows), n)               # fold the list back into rows x columns and store it in those rows
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
    D /= np.linalg.norm(D, axis=-1, keepdims=True)             # make every direction length 1 (axis=-1: the 3 coordinates)

    tri = np.asarray(triangles_CK, float)                      # tri[i] = the 3 corners of triangle i
    # for ALL triangles at once: v0 = first corner, e1 / e2 = the two edge vectors leaving that corner
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    s = S - v0                                                 # vector from every triangle's first corner to the source
    # everything that depends only on the triangle (not on the ray) is computed once per triangle;
    # a ray only enters through a dot product with the vectors in G
    # np.cross takes the cross product of each pair of rows; stacking gives, for every triangle, a 3x3 table
    G = np.stack([np.cross(e2, e1), np.cross(e2, s), np.cross(s, e1)], axis=2)   # (N, 3, 3): columns g_a, g_u, g_v/q
    # np.einsum("ij,ij->i", ...): for every triangle i, multiply e2 and G's third column coordinate by
    # coordinate and add up (a dot product, done for all triangles without a loop)
    t_num = np.einsum("ij,ij->i", e2, G[:, :, 2])                                # e2 . (s x e1)

    # a triangle can only be hit by rays whose pixel lies inside the projection of its corners, so we find
    # that pixel box first; this turns 76 426 triangles x 4 million rays into many small loops
    half = config.DETECTOR_SIZE_mm / 2
    pix = (project_to_detector(tri.reshape(-1, 3), pose) + half) / pixel_mm      # projected corners in pixel units
    pix = pix.reshape(-1, 3, 2)                                  # back to: triangle, corner (3), (column, row)
    # for every triangle: smallest / largest column and row among its 3 corners (min/max over the corners),
    # rounded down to whole pixels, widened by 1 pixel for safety, and clipped (np.clip) to 0 ... n-1
    # so the box never leaves the image
    k0 = np.clip(np.floor(pix[:, :, 0].min(axis=1)).astype(int) - 1, 0, n - 1)
    k1 = np.clip(np.floor(pix[:, :, 0].max(axis=1)).astype(int) + 1, 0, n - 1)
    j0 = np.clip(np.floor(pix[:, :, 1].min(axis=1)).astype(int) - 1, 0, n - 1)
    j1 = np.clip(np.floor(pix[:, :, 1].max(axis=1)).astype(int) + 1, 0, n - 1)
    # True for triangles whose projection lies completely beside the detector (no ray can hit them)
    outside = ((pix[:, :, 0].max(axis=1) < 0) | (pix[:, :, 0].min(axis=1) > n)
               | (pix[:, :, 1].max(axis=1) < 0) | (pix[:, :, 1].min(axis=1) > n))

    length = np.zeros((n, n))
    balance = np.zeros((n, n), dtype=int)       # entries minus exits per ray: a consistency check
    # ~outside = "not outside"; flatnonzero gives the numbers of the triangles that are still in play
    for i in np.flatnonzero(~outside):
        d = D[j0[i]:j1[i] + 1, k0[i]:k1[i] + 1]                 # directions of only the rays in this triangle's pixel box
        # d @ G[i] = dot product of every ray direction with the three vectors of this triangle;
        # moveaxis then splits the 3 results into three tables A, B, C (one value per ray)
        A, B, C = np.moveaxis(d @ G[i], -1, 0)                  # a, d.(e2 x s), d.(s x e1)
        # A can be 0 for a ray parallel to the triangle; dividing then gives inf / nan, which is fine
        # because those rays are thrown out by the "hit" test below, so we silence the warning
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v, t = B / A, C / A, t_num[i] / A
        # True / False table over the box: ray not parallel, and its hit point is inside the triangle
        # (u, v >= 0 and u + v <= 1) and in front of the source (t > 0). "&" means "and", pixel by pixel
        hit = (np.abs(A) > 1e-14) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        # +1 where the ray leaves the object, -1 where it enters, 0 where it misses
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
    l_bone, bal_bone = mesh_path_length(bone_triangles_CK, pose, pixel_mm)   # d for the bone surface
    l_balls = np.zeros_like(l_bone)                  # d for the three STL balls (zeros, same size as l_bone; the balls are added in)
    bal_balls = np.zeros_like(bal_bone)
    for tri in ball_triangles_CK_list:
        l, b = mesh_path_length(tri, pose, pixel_mm)
        l_balls += l; bal_balls += b
    sum_dx = (config.BONE_DENSITY_g_cm3 * l_bone + config.STEEL_DENSITY_g_cm3 * l_balls) / 10.0
    return sum_dx, {"l_bone": l_bone, "l_balls": l_balls, "bal_bone": bal_bone, "bal_balls": bal_balls}
