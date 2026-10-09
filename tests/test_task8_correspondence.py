"""
Task 8 test: Marker Correspondences, both methods (REM and inter-marker distances).

Assignment: test with the true projections, then with markers swapped on one detector.

Why these checks, and what we predict before running:
 1. True projections (Task 4), correct order: the best pairing is the identity (0, 1, 2) for both
    methods, the REM of every pair is ~0 and the reconstruction gives the true CK markers.
 2. All 6 orders of detector B's points (this includes every single swap of two markers): the
    pairing found must be the one that undoes the shuffle, i.e. perm[i] = position in the shuffled
    list of B's true point i. Both methods must agree.
 3. A wrong pairing must look wrong: the smallest cost of any wrong pairing is predicted to be
    > 5 mm for method 1 (sum of 3 REMs; rays of different markers pass each other by several mm)
    and > 5 mm for method 2 (sum of 3 distance errors).
 4. Method 2 also names the markers: with detector A in the true order, the labels are (0, 1, 2).
 5. Points found by Task 6 in the DRRs (they come in an arbitrary order and carry small errors):
    both methods must still find the true pairing (known from the nearest true marker of every found
    point). Predicted REM < 0.05 mm (sphere) and < 0.3 mm (vertebra); the reconstructed markers
    must lie within 0.5 mm of the true CK markers. For the vertebra the known distances come from
    the table while the STL balls are 0.3 - 0.7 mm off, so method 2's cost is about 1 mm there.

Run from the A1 folder:   python tests/test_task8_correspondence.py

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from correspondence import PERMUTATIONS, correspond, match_by_rem, match_by_distances
from markers import load_markers

report, n_failed = [], 0


def check(name, ok, detail=""):
    global n_failed
    n_failed += (not ok)
    report.append(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


true_M = load_markers()["CK"]
proj = np.load(os.path.join(config.OUTPUT_DIR, "markers_projected_task4.npz"))
A, B = proj["DetA_uv_mm"], proj["DetB_uv_mm"]

# 1. true projections, correct order
r = correspond(A, B)
check("true order: both methods give the identity pairing", r["rem"]["perm"] == (0, 1, 2) and r["distance"]["perm"] == (0, 1, 2))
check("true order: REM ~ 0 and markers reconstructed", np.all(r["REM"] < 1e-9) and np.allclose(r["markers_CK"], true_M, atol=1e-9),
      f"(REM {r['REM']})")

# 2.-4. every order of B
for order in PERMUTATIONS:
    B_shuffled = B[list(order)]                   # B_shuffled[q] = true point order[q]
    expected = tuple(order.index(i) for i in range(3))   # A point i (true marker i) sits at position order.index(i)
    r = correspond(A, B_shuffled)
    check(f"B order {order}: both methods find {expected}",
          r["rem"]["perm"] == expected and r["distance"]["perm"] == expected, f"(found {r['rem']['perm']} / {r['distance']['perm']})")
    wrong = [i for i, p in enumerate(PERMUTATIONS) if p != expected]
    c1 = np.sort(r["rem"]["cost"])[1]
    c2 = np.sort(r["distance"]["cost"])[1]
    check(f"B order {order}: every wrong pairing costs > 5 mm", c1 > 5 and c2 > 5, f"(smallest wrong: REM sum {c1:.2f} mm, distance error {c2:.2f} mm)")
    check(f"B order {order}: method 2 labels the markers 0, 1, 2", r["distance"]["labels"] == (0, 1, 2))

# 5. points localized in the DRRs (Task 6)
loc = np.load(os.path.join(config.OUTPUT_DIR, "markers_localized_task6.npz"))
stl = np.load(os.path.join(config.OUTPUT_DIR, "markers_stl_projected.npz"))
cases = {"sphere": (proj, true_M, 0.05), "vertebra": (stl, stl["CK"], 0.3)}
for phase, (truth, true_CK, rem_tol) in cases.items():
    uvA, uvB = loc[f"{phase}_A_uv_mm"], loc[f"{phase}_B_uv_mm"]
    nearest = lambda uv, key: np.linalg.norm(uv[:, None, :] - truth[key][None, :, :], axis=2).argmin(axis=1)
    nA, nB = nearest(uvA, "DetA_uv_mm"), nearest(uvB, "DetB_uv_mm")      # true marker of every found point
    expected = tuple(int(np.where(nB == nA[i])[0][0]) for i in range(3))
    r = correspond(uvA, uvB)
    check(f"{phase} (Task 6 points): both methods find the true pairing {expected}",
          r["rem"]["perm"] == expected and r["distance"]["perm"] == expected, f"(found {r['rem']['perm']} / {r['distance']['perm']})")
    mre = np.linalg.norm(r["markers_CK"] - true_CK[nA], axis=1)
    check(f"{phase} (Task 6 points): REM < {rem_tol} mm and markers within 0.5 mm", np.all(r["REM"] < rem_tol) and np.all(mre < 0.5),
          f"(REM {np.round(r['REM'], 4)} mm, MRE {np.round(mre, 3)} mm, method 2 cost {r['distance']['cost'].min():.2f} mm)")

report.append(f"\nNumber of failed checks: {n_failed}  ->  "
              f"{'ALL CHECKS PASSED' if n_failed == 0 else 'SOME CHECKS FAILED, see FAIL above'}")
text = "\n".join(report)
print(text)
with open(os.path.join(config.OUTPUT_DIR, "task8_correspondence_tests.txt"), "w") as f:
    f.write(text + "\n")
