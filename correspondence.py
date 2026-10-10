"""
correspondence.py - Task 8: Marker Correspondences.

The markers are identical, so on each detector we only get N points (N = 3 here), in
no particular order. We must decide which point on detector A and which point on
detector B are images of the same marker. There are N! ways to pair the points of A
with the points of B ("pairing" = permutation `perm`: point i on A goes with point
perm[i] on B). Both methods try every pairing, so they work for any (small) N.

Method 1, epipolar lines. The two X-ray sources S_A and S_B and one marker M span a
plane, and the images of M on both detectors lie in this plane (they are where the
plane cuts the detectors: the epipolar lines). All such planes contain the line
S_A S_B (the baseline), so a plane is described by one number: how far it is turned
around the baseline (its tilt, in degrees). Every point on a detector has a tilt (the
tilt of the plane through the baseline and its ray). Two points are images of the same
marker only if their tilts are equal. Cost of a pairing = sum of the N tilt
differences (degrees). The pairing with the smallest cost wins. This needs no 3-D
reconstruction and no knowledge of the markers.

Method 2, residual error (REM). Reconstruct every pair of a pairing with
reconstruct.reconstruct_point. Two rays that really come from one marker meet
(REM ~ 0); rays from two different markers usually pass each other by many mm.
Cost of a pairing = sum of the N REMs (mm). The pairing with the smallest cost wins.

The markers are identical spheres, so nothing in the images tells them apart, and
neither method names M1, M2, M3: they only give the pairing.
Both methods can fail only when a wrong pairing also fits (the ambiguity of Task 9:
markers in one plane through S_A and S_B, which then have the same tilt), so
`correspond` also returns the margin second-best cost / best cost.

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import itertools
import numpy as np
import frames
from reconstruct import reconstruct_point, back_projection_line

PERMUTATIONS = list(itertools.permutations(range(3)))     # 6 pairings of 3 markers


def permutations_of(n):
    """All n! pairings of n markers; for n = 3 this is PERMUTATIONS."""
    return PERMUTATIONS if n == 3 else list(itertools.permutations(range(n)))


def reconstruct_pairing(uv_A, uv_B, perm):
    """Reconstruct the N pairs (A point i, B point perm[i]). Returns (markers (N, 3) in CK, REM (N,))."""
    # Task 7 module: intersect the two back-projection lines of each pair; REM = how far the lines miss each other
    results = [reconstruct_point(uv_A[i], uv_B[perm[i]]) for i in range(len(perm))]
    return np.array([r[0] for r in results]), np.array([r[1] for r in results])


def plane_tilt_deg(uv, pose):
    """
    Tilt (0 to 180 degrees) of the plane through the baseline S_A S_B and the ray of the
    point `uv` (mm) on the detector of `pose`. Points that belong to the same marker have the same tilt.
    """
    baseline = frames.detector_geometry("B")["S_CK"] - frames.detector_geometry("A")["S_CK"]
    b = baseline / np.linalg.norm(baseline)
    # two directions e1, e2 that are perpendicular to the baseline and to each other: they span the
    # flat cross-section "looking along the baseline", in which a plane through the baseline is a line
    z = np.array([0.0, 0.0, 1.0])                     # any direction that is not along the baseline
    e2 = z - (z @ b) * b                              # z with its baseline part removed
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(b, e2)
    _, to_source = back_projection_line(uv, pose)     # the ray of the point (pointing at the source)
    # the angle of the ray in that cross-section is the tilt; a plane has no front or back, so mod 180
    return float(np.degrees(np.arctan2(to_source @ e2, to_source @ e1)) % 180.0)


def match_by_epipolar(uv_A, uv_B):
    """
    Method 1. Returns dict: perm (best pairing), cost (N!,) sum of tilt differences (degrees) per pairing in
    permutations_of(N) order, tilt_A / tilt_B (N,) the tilt of every point on A / B.
    """
    tilt_A = np.array([plane_tilt_deg(p, "A") for p in uv_A])
    tilt_B = np.array([plane_tilt_deg(p, "B") for p in uv_B])
    perms = permutations_of(len(uv_A))
    # difference of two tilts; 0 and 180 degrees are the same plane, so the largest difference is 90
    gap = lambda a, b: abs((a - b + 90.0) % 180.0 - 90.0)
    cost = np.array([sum(gap(tilt_A[i], tilt_B[p[i]]) for i in range(len(p))) for p in perms])
    return {"perm": perms[int(cost.argmin())], "cost": cost, "tilt_A": tilt_A, "tilt_B": tilt_B}


def match_by_rem(uv_A, uv_B):
    """Method 2. Returns dict: perm (best pairing), cost (N!,) sum of REM per pairing in permutations_of(N) order."""
    # lines from the same marker meet (REM ~ 0), lines from different markers miss by several mm,
    # so the right pairing is the one with the smallest total miss distance
    perms = permutations_of(len(uv_A))
    cost = np.array([reconstruct_pairing(uv_A, uv_B, p)[1].sum() for p in perms])
    return {"perm": perms[int(cost.argmin())], "cost": cost}


def margin(cost):
    """second-best cost / best cost: large = clear decision, near 1 = ambiguous."""
    s = np.sort(cost)
    return s[1] / max(s[0], 1e-12)


def correspond(uv_A, uv_B):
    """
    Run both methods on the N unordered points of each detector ((N, 2) mm arrays).
    Returns dict: epipolar / rem (the two method results), agree (same pairing?), perm (the
    pairing to use = method 1), markers_CK (N, 3) reconstructed with it, REM (N,).
    """
    uv_A, uv_B = np.asarray(uv_A, float), np.asarray(uv_B, float)
    r1, r2 = match_by_epipolar(uv_A, uv_B), match_by_rem(uv_A, uv_B)   # two ways to test "do these two rays meet"
    M, rem = reconstruct_pairing(uv_A, uv_B, r1["perm"])
    return {"epipolar": r1, "rem": r2, "agree": r1["perm"] == r2["perm"],
            "perm": r1["perm"], "markers_CK": M, "REM": rem}
