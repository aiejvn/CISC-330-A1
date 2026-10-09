"""
correspondence.py - Task 8: Marker Correspondences.

The markers are identical, so on each detector we only get three points, in no
particular order. We must decide which point on detector A and which point on
detector B are images of the same marker. There are 3! = 6 ways to pair the
three points of A with the three points of B ("pairing" = permutation `perm`:
point i on A goes with point perm[i] on B).

Method 1, residual error (REM). Reconstruct every pair of a pairing with
reconstruct.reconstruct_point. Two rays that really come from one marker meet
(REM ~ 0); rays from two different markers usually pass each other by many mm.
Cost of a pairing = sum of the 3 REMs. The pairing with the smallest cost wins.

Method 2, inter-marker distances. The markers are fixed on a rigid vertebra, so
the distances between them are known (from the marker table). Reconstruct all
three markers of a pairing and compare the three distances between them with
the three known distances. A wrong pairing produces three points that are
somewhere else, so their mutual distances do not match. Cost of a pairing =
sum over the 3 marker pairs of | reconstructed distance - known distance |, with
the best naming of the reconstructed points M1, M2, M3 (this also tells which
marker is which). The pairing with the smallest cost wins.

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

PERMUTATIONS = list(itertools.permutations(range(3)))     # 6 pairings


def reference_distances(markers_mm=None):
    """(3, 3) matrix of the known distances between the markers (mm); default: the assignment's marker table."""
    m = np.asarray(config.MARKERS_MD_mm if markers_mm is None else markers_mm, float)
    return np.linalg.norm(m[:, None, :] - m[None, :, :], axis=2)


def reconstruct_pairing(uv_A, uv_B, perm):
    """Reconstruct the 3 pairs (A point i, B point perm[i]). Returns (markers (3, 3) in CK, REM (3,))."""
    results = [reconstruct_point(uv_A[i], uv_B[perm[i]]) for i in range(3)]
    return np.array([r[0] for r in results]), np.array([r[1] for r in results])


def match_by_rem(uv_A, uv_B):
    """Method 1. Returns dict: perm (best pairing), cost (6,) sum of REM per pairing in PERMUTATIONS order."""
    cost = np.array([reconstruct_pairing(uv_A, uv_B, p)[1].sum() for p in PERMUTATIONS])
    return {"perm": PERMUTATIONS[int(cost.argmin())], "cost": cost}


def match_by_distances(uv_A, uv_B, ref_dist=None):
    """
    Method 2. Returns dict: perm (best pairing), labels (labels[i] = marker index 0..2 of the
    reconstructed point i of the best pairing), cost (6,) per pairing in PERMUTATIONS order.
    """
    ref = reference_distances() if ref_dist is None else ref_dist
    pairs = [(0, 1), (0, 2), (1, 2)]
    cost, best_labels = [], []
    for p in PERMUTATIONS:
        M, _ = reconstruct_pairing(uv_A, uv_B, p)
        d = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=2)
        per_label = [sum(abs(d[i, j] - ref[l[i], l[j]]) for i, j in pairs) for l in PERMUTATIONS]
        cost.append(min(per_label))
        best_labels.append(PERMUTATIONS[int(np.argmin(per_label))])
    k = int(np.argmin(cost))
    return {"perm": PERMUTATIONS[k], "labels": best_labels[k], "cost": np.array(cost)}


def margin(cost):
    """second-best cost / best cost: large = clear decision, near 1 = ambiguous."""
    s = np.sort(cost)
    return s[1] / max(s[0], 1e-12)


def correspond(uv_A, uv_B):
    """
    Run both methods on the three unordered points of each detector ((3, 2) mm arrays).
    Returns dict: rem / distance (the two method results), agree (same pairing?), perm (the
    pairing to use = method 1), markers_CK (3, 3) reconstructed with it, REM (3,).
    """
    uv_A, uv_B = np.asarray(uv_A, float), np.asarray(uv_B, float)
    r1, r2 = match_by_rem(uv_A, uv_B), match_by_distances(uv_A, uv_B)
    M, rem = reconstruct_pairing(uv_A, uv_B, r1["perm"])
    return {"rem": r1, "distance": r2, "agree": r1["perm"] == r2["perm"],
            "perm": r1["perm"], "markers_CK": M, "REM": rem}
