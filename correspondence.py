"""
correspondence.py - Task 8: Marker Correspondences.

The markers are identical, so on each detector we only get N points (N = 3 here), in
no particular order. We must decide which point on detector A and which point on
detector B are images of the same marker. There are N! ways to pair the points of A
with the points of B ("pairing" = permutation `perm`: point i on A goes with point
perm[i] on B). Both methods try every pairing, so they work for any (small) N.

Method 1, residual error (REM). Reconstruct every pair of a pairing with
reconstruct.reconstruct_point. Two rays that really come from one marker meet
(REM ~ 0); rays from two different markers usually pass each other by many mm.
Cost of a pairing = sum of the N REMs. The pairing with the smallest cost wins.

Method 2, rigid fit of the marker model. The markers are fixed on a rigid vertebra,
so the reconstructed points of the right pairing must be the marker table moved by a
rotation and a translation. For every pairing we reconstruct the N points, fit the
marker table to them (Kabsch/SVD, proper rotation only) for every naming of the points
and take the RMS residual (fiducial registration error, mm). Cost of a pairing = the
smallest residual over the namings. The pairing with the smallest cost wins, and the
best naming tells which marker is which. This uses the whole shape of the marker set,
not only some of its distances, so it does not depend on this particular arrangement.
If the markers are symmetric (e.g. equidistant), the pairing is still found but several
namings fit equally well: `label_cost` shows this.

Both methods can fail only when a wrong pairing also has near-zero REM
(the ambiguity of Task 9: markers in one plane through S_A and S_B), so
`correspond` also returns the margin second-best cost / best cost.

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import itertools
import numpy as np
import config
from reconstruct import reconstruct_point

PERMUTATIONS = list(itertools.permutations(range(3)))     # 6 pairings of 3 markers


def permutations_of(n):
    """All n! pairings (or namings) of n markers; for n = 3 this is PERMUTATIONS."""
    return PERMUTATIONS if n == 3 else list(itertools.permutations(range(n)))


def reconstruct_pairing(uv_A, uv_B, perm):
    """Reconstruct the N pairs (A point i, B point perm[i]). Returns (markers (N, 3) in CK, REM (N,))."""
    results = [reconstruct_point(uv_A[i], uv_B[perm[i]]) for i in range(len(perm))]
    return np.array([r[0] for r in results]), np.array([r[1] for r in results])


def match_by_rem(uv_A, uv_B):
    """Method 1. Returns dict: perm (best pairing), cost (N!,) sum of REM per pairing in permutations_of(N) order."""
    perms = permutations_of(len(uv_A))
    cost = np.array([reconstruct_pairing(uv_A, uv_B, p)[1].sum() for p in perms])
    return {"perm": perms[int(cost.argmin())], "cost": cost}


def rigid_fit_error(model, points):
    """
    RMS residual (mm) after the best proper rigid motion (rotation + translation) of `model`
    (N, 3) onto `points` (N, 3): Kabsch with the SVD of the covariance of the centered sets.
    """
    P, Q = model - model.mean(axis=0), points - points.mean(axis=0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])      # det = +1: no mirror image
    R = Vt.T @ D @ U.T
    return float(np.sqrt(np.mean(np.sum((P @ R.T - Q) ** 2, axis=1))))


def match_by_fit(uv_A, uv_B, model_mm=None):
    """
    Method 2. `model_mm` (N, 3) is the known marker set (default: the assignment's marker table).
    Returns dict: perm (best pairing), labels (labels[i] = marker index of the reconstructed point i
    of the best pairing), cost (N!,) per pairing in permutations_of(N) order, label_cost (N!,) the
    fit error of every naming for the best pairing (several near 0 = symmetric markers).
    """
    model = np.asarray(config.MARKERS_MD_mm if model_mm is None else model_mm, float)
    perms = permutations_of(len(model))
    cost, label_costs = [], []
    for p in perms:
        M, _ = reconstruct_pairing(uv_A, uv_B, p)
        per_label = np.array([rigid_fit_error(model[list(l)], M) for l in perms])
        cost.append(per_label.min())
        label_costs.append(per_label)
    k = int(np.argmin(cost))
    return {"perm": perms[k], "labels": perms[int(np.argmin(label_costs[k]))],
            "cost": np.array(cost), "label_cost": label_costs[k]}


def margin(cost):
    """second-best cost / best cost: large = clear decision, near 1 = ambiguous."""
    s = np.sort(cost)
    return s[1] / max(s[0], 1e-12)


def correspond(uv_A, uv_B, model_mm=None):
    """
    Run both methods on the N unordered points of each detector ((N, 2) mm arrays).
    Returns dict: rem / fit (the two method results), agree (same pairing?), perm (the
    pairing to use = method 1), markers_CK (N, 3) reconstructed with it, REM (N,).
    """
    uv_A, uv_B = np.asarray(uv_A, float), np.asarray(uv_B, float)
    r1, r2 = match_by_rem(uv_A, uv_B), match_by_fit(uv_A, uv_B, model_mm)
    M, rem = reconstruct_pairing(uv_A, uv_B, r1["perm"])
    return {"rem": r1, "fit": r2, "agree": r1["perm"] == r2["perm"],
            "perm": r1["perm"], "markers_CK": M, "REM": rem}
