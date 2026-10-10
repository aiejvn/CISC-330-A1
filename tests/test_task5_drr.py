"""
Task 5 test: DRR, sphere phase. Coarse detector (0.5 mm pixels, 400 x 400) so it runs fast.

Why these checks, and what we predict before running:
 1. Chord formula: a ray through the sphere center has length 2R = 100 mm; a ray at
    distance h from the center has length 2 sqrt(R^2 - h^2); a ray at h > R has 0.
 2. Pixel rays: the middle of pixel (k, j) must agree with F_CK_from_Det @ F_Det_from_Img
    (the route written in the README); and the ray of a pixel must hit the plane w = 0.
 3. All markers lie inside the sphere (|M| + 1.5 < 50; the 3 mm balls).
 4. Corner pixel (u, v) = (-99.75, -99.75): its ray misses the sphere -> DRR value 0 exactly.
 5. Sphere-only pixel value: 0.25 * chord/10 with chord from the cross-product
    distance h = |(c - S) x d|  (a different formula than the quadratic).
 6. Marker contrast: at the pixel containing a marker's true projection (Task 4), the
    ray passes within ~0.2 mm of the ball center, so the ball chord is ~3 mm and
        DRR(with) - DRR(without) = (7.85 - 0.25) * 0.3 = 2.28 g/cm^2.
    Predicted range 2.0 - 2.4. This also catches a transposed image (img[j, k] vs img[k, j]).
 7. Away from the markers the images with and without markers are identical.

Run from the A1 folder:   python tests/test_task5_drr.py

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, frames
from geometry import line_sphere_chord, apply_transform
from projector import project_to_detector
from markers import load_markers
import drr

PIXEL_MM = 0.5
M_CK = load_markers()["CK"]
report, n_failed = [], 0


def check(name, ok, detail=""):
    global n_failed
    n_failed += (not ok)
    report.append(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# 1. chord formula
R = config.SPHERE_RADIUS_mm
S_A = frames.detector_geometry("A")["S_CK"]
to_center = -S_A / np.linalg.norm(S_A)
check("chord through the center = 2R", abs(line_sphere_chord(S_A, to_center, np.zeros(3), R) - 2 * R) < 1e-9)
# a ray parallel to the central one but shifted sideways by h = 30 mm (shift along +z)
h = 30.0
S_shift = S_A + np.array([0, 0, h])
check("chord at distance 30 mm = 2 sqrt(R^2 - 900)",
      abs(line_sphere_chord(S_shift, to_center, np.zeros(3), R) - 2 * np.sqrt(R**2 - h**2)) < 1e-9)
S_far = S_A + np.array([0, 0, 60.0])
check("chord at distance 60 mm > R is 0", line_sphere_chord(S_far, to_center, np.zeros(3), R) == 0.0)

# 2. pixel rays
for pose in ("A", "B"):
    k, j = 123, 301
    mid = drr.pixel_centers_CK(pose, PIXEL_MM, [j])[0, k]
    uv = np.array([(k + 0.5) * PIXEL_MM - 100, (j + 0.5) * PIXEL_MM - 100, 0.0])
    expected = apply_transform(frames.F_CK_from_Det(pose), uv)
    # README route in image pixels of the coarse image: i1 = (k+0.5)*pixel_mm/0.1 (full-res pixel units)
    i = np.array([(k + 0.5) * PIXEL_MM / config.PIXEL_SIZE_mm, (j + 0.5) * PIXEL_MM / config.PIXEL_SIZE_mm, 0.0])
    via_readme = apply_transform(frames.F_CK_from_Det(pose), apply_transform(frames.F_Det_from_Img(), i))
    check(f"pixel middle pose {pose}: direct == via F_Det_from_Img", np.allclose(mid, expected) and np.allclose(mid, via_readme))
    w = frames.detector_geometry(pose)
    check(f"pixel middle pose {pose} lies in the detector plane", abs((mid - w['C_CK']) @ w['w']) < 1e-9)

# 3. markers inside
reach = np.linalg.norm(M_CK, axis=1) + drr.MARKER_DIAMETER_mm / 2
check("all marker balls inside the sphere", np.all(reach < R), f"reach = {np.round(reach, 3)} mm")

for pose in ("A", "B"):
    img_with = drr.render_sphere_drr(pose, PIXEL_MM, M_CK, with_markers=True)
    img_without = drr.render_sphere_drr(pose, PIXEL_MM, M_CK, with_markers=False)
    n = img_with.shape[0]
    check(f"pose {pose}: image shape {n} x {n}", img_with.shape == (400, 400))

    # 4. corner pixel misses the sphere
    check(f"pose {pose}: corner pixel value == 0", img_without[0, 0] == 0.0 and img_with[0, 0] == 0.0)

    # 5. sphere-only value from the cross-product distance
    det = frames.detector_geometry(pose)
    for (k, j) in [(200, 200), (150, 260), (320, 90)]:
        P = drr.pixel_centers_CK(pose, PIXEL_MM, [j])[0, k]
        d = (P - det["S_CK"]) / np.linalg.norm(P - det["S_CK"])
        h = np.linalg.norm(np.cross(-det["S_CK"], d))               # distance center-line, (c - S) x d with c = 0
        chord = 2 * np.sqrt(max(R**2 - h**2, 0.0))
        pred = config.SPHERE_DENSITY_g_cm3 * chord / 10
        check(f"pose {pose}: sphere-only pixel (k={k}, j={j})", abs(img_without[j, k] - pred) < 1e-12,
              f"image {img_without[j, k]:.6f}  predicted {pred:.6f}")

    # 6. marker contrast at the true projection
    uv = project_to_detector(M_CK, pose)
    for name, (u, v) in zip(config.MARKER_NAMES, uv):
        k, j = int(np.floor((u + 100) / PIXEL_MM)), int(np.floor((v + 100) / PIXEL_MM))
        bump = img_with[j, k] - img_without[j, k]
        check(f"pose {pose}: {name} contrast at pixel (k={k}, j={j})", 2.0 <= bump <= 2.4,
              f"DRR(with) - DRR(without) = {bump:.4f} g/cm^2  (predicted 2.0-2.4)")

    # 7. images identical away from the markers (outside 8 mm of every projection)
    kk, jj = np.meshgrid(np.arange(n), np.arange(n))
    far = np.ones((n, n), bool)
    for (u, v) in uv:
        far &= np.hypot((kk + 0.5) * PIXEL_MM - 100 - u, (jj + 0.5) * PIXEL_MM - 100 - v) > 8.0
    check(f"pose {pose}: images equal away from the markers", np.array_equal(img_with[far], img_without[far]))

report.append(f"\nNumber of failed checks: {n_failed}  ->  "
              f"{'ALL CHECKS PASSED' if n_failed == 0 else 'SOME CHECKS FAILED, see FAIL above'}")
text = "\n".join(report)
print(text)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
with open(os.path.join(config.OUTPUT_DIR, "task5_drr_tests.txt"), "w") as f:
    f.write(text + "\n")
