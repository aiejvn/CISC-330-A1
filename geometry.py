"""
geometry.py - small reusable geometry utilities (Math Primer Part 1 and 2).

Conventions (Math Primer Part 2):
  * points are column vectors; a 4x4 matrix acts from the left: p_new = F @ p
  * homogeneous point = [x, y, z, 1]
  * F_a_from_b means "transform FROM frame b TO frame a" (the course writes F(a<-b))
Functions accept one point with shape (3,) or many points with shape (N, 3).

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import numpy as np


def normalize(vector):
    """Return the unit vector in the direction of `vector`."""
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if length < 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / length


# ---- elementary 4x4 homogeneous matrices -------------------------------------
def translation(d):
    """Homogeneous translation T(d): moves a point by the vector d."""
    T = np.eye(4)
    T[:3, 3] = d
    return T


def rotation_x(angle_deg):
    """Homogeneous rotation about +x by angle_deg (right-hand rule)."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.eye(4)
    R[1:3, 1:3] = [[c, -s], [s, c]]
    return R


def rotation_y(angle_deg):
    """Homogeneous rotation about +y by angle_deg (right-hand rule)."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.eye(4)
    R[0, 0], R[0, 2], R[2, 0], R[2, 2] = c, s, -s, c
    return R


def rotation_z(angle_deg):
    """Homogeneous rotation about +z by angle_deg (right-hand rule: +x turns toward +y)."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.eye(4)
    R[0:2, 0:2] = [[c, -s], [s, c]]
    return R


def scaling(factor):
    """Homogeneous uniform scaling (used for mm -> pixel conversion)."""
    S = np.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = factor
    return S


# ---- frames ------------------------------------------------------------------
def frame_to_home(O_e, e1, e2, e3):
    """
    F_home_from_e: the transformation from a frame e to the home frame, where
    the frame's center O_e and unit base vectors e1, e2, e3 are written in home
    coordinates. Math Primer Part 2: "to home, base vectors in COLUMNS"; the
    center O_e goes in the last column (first rotate, then translate by O_e).
    """
    F = np.eye(4)
    F[:3, 0], F[:3, 1], F[:3, 2] = e1, e2, e3
    F[:3, 3] = O_e
    return F


def invert_rigid(F):
    """
    Inverse of a rigid body transformation (rotation R followed by translation t):
        F^-1 = [R^T, -R^T t]
    Math Primer Part 2: "from home, base vectors in ROWS" (R^T), after
    translating by -O_e. Using R^T instead of a general matrix inverse works
    because R^T R = I for a rotation.
    """
    R, t = F[:3, :3], F[:3, 3]
    F_inv = np.eye(4)
    F_inv[:3, :3] = R.T
    F_inv[:3, 3] = -R.T @ t
    return F_inv


def apply_transform(F, points):
    """Apply a 4x4 transform to one point (3,) or many points (N, 3)."""
    points = np.asarray(points, dtype=float)
    single = points.ndim == 1                      # remember if we were given just one point
    P = np.atleast_2d(points)                      # make it a table: one point per row, shape (N, 3)
    P_h = np.hstack([P, np.ones((len(P), 1))])     # add the 4th coordinate "1" -> homogeneous (N, 4)
    # F acts on COLUMN vectors, so turn the rows into columns (.T), multiply,
    # turn back into rows, and drop the 4th coordinate again.
    Q = (F @ P_h.T).T[:, :3]
    return Q[0] if single else Q


def is_rigid(F, tol=1e-9):
    """
    True if the 3x3 part of F is a rotation matrix: R^T R = I (orthonormal
    base vectors) and det R = +1 (right-handed), so F is a rigid body transformation.
    """
    R = F[:3, :3]
    return np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol


# ---- lines and planes (Math Primer Part 1) -------------------------------------
def line_plane_intersection(P, v, A, n):
    """
    Intersect the line L = P + t*v with the plane through A with normal n.
    Math Primer: t = ((A - P) . n) / (v . n), then L = P + t*v.
    Works for one line (P, v shape (3,)) or many lines (shape (N, 3)).
    Returns (points, t).
    """
    P, v = np.asarray(P, float), np.asarray(v, float)
    A, n = np.asarray(A, float), np.asarray(n, float)
    v_dot_n = v @ n
    if np.any(np.abs(v_dot_n) < 1e-12):
        raise ValueError("line is parallel to the plane: no intersection")
    t = ((A - P) @ n) / v_dot_n                    # one number t per line
    # With many lines, t has shape (N,) but v has shape (N, 3); t[..., None]
    # turns t into shape (N, 1) so that each line's t multiplies its own v.
    points = P + (t[..., None] if np.ndim(t) else t) * v
    return points, t


def line_sphere_chord(S, d, center, radius):
    """
    Length of the part of the lines S + t*d (t real) that lies inside a sphere.
    d = unit directions, shape (3,) or (N, 3); S, center = (3,). With m = S - center,
        |m + t d|^2 = R^2  ->  t^2 + 2 (m.d) t + (m.m - R^2) = 0
    b = m.d, disc = b^2 - (m.m - R^2); roots t = -b +- sqrt(disc); chord = 2 sqrt(disc)
    (0 if disc <= 0, i.e. the line misses the sphere). Needs |d| = 1.
    """
    S, center = np.asarray(S, float), np.asarray(center, float)
    d = np.asarray(d, float)
    m = S - center
    b = d @ m
    disc = b * b - (m @ m - radius ** 2)
    return 2.0 * np.sqrt(np.maximum(disc, 0.0))


def symbolic_intersection(P1, v1, P2, v2):
    """
    Symbolic intersection of two lines in 3D (Math Primer Part 1, slides
    "Symbolic intersection of lines in 3D"). The lines are
        L1 = P1 + t1*v1   and   L2 = P2 + t2*v2      (v1, v2 unit vectors)
    Two back-projection lines almost never meet exactly, so, as on the slides:
      1. look for the line L3 that is perpendicular to both lines,
         v3 = normalize(v1 x v2);
      2. find the intersection of L3 with L1 and with L2;
      3. find the middle point M on L3.
    The conditions of intersection give  P1 + t1 v1 = P2 + t2 v2 + t3 v3,  i.e.
        P1 - P2 = -t1 v1 + t2 v2 + t3 v3
    which is 3 linear equations in the 3 unknowns t1, t2, t3:
        [-v1  v2  v3] (t1, t2, t3)^T = P1 - P2
    Plug t1 and t2 back into the line equations:
        L1 = P1 + t1 v1,  L2 = P2 + t2 v2,  M = (L1 + L2) / 2,  d = |L1 - L2| = |t3|
    Updated Math Primer slide: d = REM, the distance between the two lines.

    Returns M, d, L1, L2 (L1 and L2 are the two closest points).
    """
    P1, v1, P2, v2 = (np.asarray(a, dtype=float) for a in (P1, v1, P2, v2))
    v1_cross_v2 = np.cross(v1, v2)
    if np.linalg.norm(v1_cross_v2) < 1e-12:
        raise ValueError("the two lines are parallel: no single closest pair of points")
    v3 = normalize(v1_cross_v2)
    A = np.column_stack([-v1, v2, v3])          # columns -v1, v2, v3, exactly as on the slide
    t1, t2, t3 = np.linalg.solve(A, P1 - P2)    # Gaussian elimination
    L1 = P1 + t1 * v1
    L2 = P2 + t2 * v2
    M = (L1 + L2) / 2
    d = np.linalg.norm(L1 - L2)                 # equals |t3|
    return M, d, L1, L2
