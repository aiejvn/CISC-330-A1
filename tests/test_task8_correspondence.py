"""
Task 8 test: Marker Correspondences, both methods (epipolar tilt and REM).

Assignment: test with the true projections, then with markers swapped on one detector,
and solve it with a second, different method and run the same tests.

Why these checks, and what we predict before running:
 1. True projections (Task 4), correct order: the best pairing is the identity (0, 1, 2) for both
    methods, the REM of every pair is ~0 and the reconstruction gives the true CK markers.
 2. All 6 orders of detector B's points (this includes every single swap of two markers): the
    pairing found must be the one that undoes the shuffle, i.e. perm[i] = position in the shuffled
    list of B's true point i. Both methods must agree.
 3. A wrong pairing must look wrong: the smallest cost of any wrong pairing is predicted to be
    > 1 degree for method 1 (sum of 3 tilt differences; our markers are all within about 1 degree
    of the plane through both sources, so the tilts of different markers are close, but not equal)
    and > 5 mm for method 2 (sum of 3 REMs; rays of different markers pass each other by several mm).
 4. Points found by Task 6 in the DRRs (they come in an arbitrary order and carry small errors):
    both methods must still find the true pairing (known from the nearest true marker of every found
    point). Predicted REM < 0.05 mm (sphere) and < 0.3 mm (vertebra); the reconstructed markers
    must lie within 0.5 mm of the true CK markers.
 5. Other marker arrangements (the methods must not rely on our table): (a) three equidistant
    markers (equilateral triangle): both methods still find the pairing; (b) four markers at generic
    positions: all 4! = 24 orders of detector B are undone by both methods.

Run from the A1 folder:   python tests/test_task8_correspondence.py

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from correspondence import PERMUTATIONS, permutations_of, correspond
from geometry import rotation_x, rotation_y, apply_transform
from projector import project_to_detector
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
check("true order: both methods give the identity pairing", r["epipolar"]["perm"] == (0, 1, 2) and r["rem"]["perm"] == (0, 1, 2))
check("true order: REM ~ 0 and markers reconstructed", np.all(r["REM"] < 1e-9) and np.allclose(r["markers_CK"], true_M, atol=1e-9),
      f"(REM {r['REM']})")

# 2.-3. every order of B
for order in PERMUTATIONS:
    B_shuffled = B[list(order)]                   # B_shuffled[q] = true point order[q]
    expected = tuple(order.index(i) for i in range(3))   # A point i (true marker i) sits at position order.index(i)
    r = correspond(A, B_shuffled)
    check(f"B order {order}: both methods find {expected}",
          r["epipolar"]["perm"] == expected and r["rem"]["perm"] == expected, f"(found {r['epipolar']['perm']} / {r['rem']['perm']})")
    c1 = np.sort(r["epipolar"]["cost"])[1]        # smallest cost of a wrong pairing (the best one costs ~0)
    c2 = np.sort(r["rem"]["cost"])[1]
    check(f"B order {order}: every wrong pairing costs > 1 degree (method 1) / > 5 mm (method 2)", c1 > 1 and c2 > 5,
          f"(smallest wrong: tilt difference {c1:.2f} degrees, REM sum {c2:.2f} mm)")

# 4. points localized in the DRRs (Task 6)
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
          r["epipolar"]["perm"] == expected and r["rem"]["perm"] == expected, f"(found {r['epipolar']['perm']} / {r['rem']['perm']})")
    mre = np.linalg.norm(r["markers_CK"] - true_CK[nA], axis=1)
    check(f"{phase} (Task 6 points): REM < {rem_tol} mm and markers within 0.5 mm", np.all(r["REM"] < rem_tol) and np.all(mre < 0.5),
          f"(REM {np.round(r['REM'], 4)} mm, MRE {np.round(mre, 3)} mm, "
          f"best tilt cost {r['epipolar']['cost'].min():.4f} degrees)")

# 5. other marker arrangements
def project_set(points):
    return project_to_detector(points, "A"), project_to_detector(points, "B")


tri = np.array([[0, 0, 0], [40, 0, 0], [20, 20 * np.sqrt(3), 0]], float)          # equilateral, side 40 mm
tri = apply_transform(rotation_y(25) @ rotation_x(35), tri - tri.mean(axis=0))      # tilted so the markers differ in height
uvA, uvB = project_set(tri)
r = correspond(uvA, uvB)
check("equidistant markers: both methods find the pairing", r["epipolar"]["perm"] == (0, 1, 2) and r["rem"]["perm"] == (0, 1, 2))

four = np.array([[-25, 20, 10], [30, 15, -12], [5, -30, 25], [18, 28, 31]], float)  # generic positions, different heights
uvA, uvB = project_set(four)
ok_all = True
for order in permutations_of(4):
    expected = tuple(order.index(i) for i in range(4))
    r = correspond(uvA, uvB[list(order)])
    ok_all &= (r["epipolar"]["perm"] == expected and r["rem"]["perm"] == expected)
check("4 markers: both methods undo all 24 orders of detector B", ok_all)

report.append(f"\nNumber of failed checks: {n_failed}  ->  "
              f"{'ALL CHECKS PASSED' if n_failed == 0 else 'SOME CHECKS FAILED, see FAIL above'}")
text = "\n".join(report)
print(text)
with open(os.path.join(config.OUTPUT_DIR, "task8_correspondence_tests.txt"), "w") as f:
    f.write(text + "\n")
