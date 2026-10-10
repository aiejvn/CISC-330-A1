"""
Task 5 test: DRR, vertebra phase (STL model with its own marker balls, 2 mm in the file, enlarged to 3 mm), coarse
detector (0.5 mm pixels, 400 x 400) so it runs in about a minute.

Why these checks, and what we predict before running:
 1. Mesh facts (measured before writing the code): 6 pieces; the bone is closed with
    POSITIVE signed volume (outward normals, ~68,585 mm^3); the 3 balls have volume
    ~4.15 mm^3 (a 2 mm sphere is 4.19) and diameter 2; one small piece is an internal
    cavity with NEGATIVE volume. We predict: exactly 3 balls found, matched to M1, M2, M3
    in table order, each within 0.5 mm of its table position. After enlarging each ball
    about its center to 3 mm the volume is 4.15 * 1.5^3 = 14.0 mm^3 (a 3 mm sphere: 14.14).
 2. Entry/exit balance: for a closed surface and a ray that starts outside, entries minus
    exits is 0 for EVERY pixel. Any double-counted or missed hit on an edge would break
    this. Predict: balance 0 everywhere, for bone and for each ball, in both poses.
 3. Lengths >= 0 (a negative length would mean a sign error).
 4. Independent check: for 300 random pixels, brute force against ALL triangles with
    the classic method (collect hits, sort by t, pair entry/exit: length = sum(t_exit - t_entry)),
    no bounding boxes, no orientation. Predict: equal to the fast method within 1e-9 mm.
 5. Ball chord: at the pixel containing the projected STL ball centre, the ray passes
    within ~0.2 mm of the centre of a 3 mm ball whose mesh is inscribed (642 corners),
    so the chord is slightly under 3 mm: predict 2.85 - 3.0 mm for every ball and pose.
    (This also catches a transposed image: img[j, k] vs img[k, j].)
 6. Pixel value: at that pixel the DRR = (x_bone l_bone + x_steel l_balls)/10 is reproduced.

Run from the A1 folder:   python tests/test_task5_vertebra.py

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, frames, drr
from stl_io import read_binary_stl, split_vertebra_and_markers, signed_volume, enlarge_ball
from geometry import apply_transform
from projector import project_to_detector

PIXEL_MM = 0.5
report, n_failed = [], 0


def check(name, ok, detail=""):
    global n_failed
    n_failed += (not ok)
    report.append(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def to_CK(tri_MD):
    return apply_transform(frames.F_CK_from_MD(), tri_MD.reshape(-1, 3)).reshape(-1, 3, 3)


def brute_force_length(tri, S, d):
    """Classic method for ONE ray: all hits, sorted by t, paired (entry, exit)."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-14
    f = 1.0 / np.where(ok, a, 1.0)
    s = S - tri[:, 0]
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * (q @ d)
    t = f * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    ts = np.sort(t[hit])
    assert len(ts) % 2 == 0, "odd number of hits"
    return np.sum(ts[1::2] - ts[0::2])


# 1. mesh facts
triangles = read_binary_stl(config.STL_PATH)
bone_MD, balls_MD, centers_MD = split_vertebra_and_markers(triangles, config.MARKERS_MD_mm)
check("3 balls found", len(balls_MD) == 3)
check("bone group has positive signed volume (outward normals)", signed_volume(bone_MD) > 60000,
      f"volume {signed_volume(bone_MD):.1f} mm^3")
check("ball volumes ~ 4.15 mm^3 (2 mm sphere)", all(abs(signed_volume(b) - 4.15) < 0.1 for b in balls_MD),
      str([round(signed_volume(b), 3) for b in balls_MD]))
dist = np.linalg.norm(centers_MD - config.MARKERS_MD_mm, axis=1)
check("each ball within 0.5 mm of its table position (order M1, M2, M3)", np.all(dist < 0.5), f"{np.round(dist, 3)} mm")
check("balls + bone cover every triangle", len(bone_MD) + sum(len(b) for b in balls_MD) == len(triangles))
# the balls of the render: enlarged about their own centers to 3 mm (same as run_task5_drr.py)
balls_MD = [enlarge_ball(b, c, config.MARKER_DIAMETER_mm) for b, c in zip(balls_MD, centers_MD)]
check("enlarged ball volumes ~ 14.0 mm^3 (3 mm sphere is 14.14)", all(abs(signed_volume(b) - 14.0) < 0.3 for b in balls_MD),
      str([round(signed_volume(b), 3) for b in balls_MD]))

bone = to_CK(bone_MD)
balls = [to_CK(b) for b in balls_MD]
centers_CK = apply_transform(frames.F_CK_from_MD(), centers_MD)

rng = np.random.RandomState(0)
for pose in ("A", "B"):
    img, info = drr.render_vertebra_drr(pose, PIXEL_MM, bone, balls)
    # 2. balance
    check(f"pose {pose}: entry/exit balance 0 for the bone, every pixel", np.all(info["bal_bone"] == 0),
          f"{int(np.count_nonzero(info['bal_bone']))} bad pixels")
    check(f"pose {pose}: entry/exit balance 0 for the balls, every pixel", np.all(info["bal_balls"] == 0),
          f"{int(np.count_nonzero(info['bal_balls']))} bad pixels")
    # 3. non-negative
    check(f"pose {pose}: all lengths >= 0", info["l_bone"].min() >= -1e-9 and info["l_balls"].min() >= -1e-9,
          f"min bone {info['l_bone'].min():.2e}, min balls {info['l_balls'].min():.2e}")
    check(f"pose {pose}: bone is in the image (max chord {info['l_bone'].max():.1f} mm)", info["l_bone"].max() > 20)

    # 4. brute force at random pixels (half of them chosen among pixels where the bone was hit)
    S = frames.detector_geometry(pose)["S_CK"]
    n = img.shape[0]
    hit_pixels = np.argwhere(info["l_bone"] > 0)
    pick = np.vstack([hit_pixels[rng.randint(len(hit_pixels), size=150)], rng.randint(n, size=(150, 2))])
    worst = 0.0
    for j, k in pick:
        P = drr.pixel_centers_CK(pose, PIXEL_MM, [j])[0, k]
        d = (P - S) / np.linalg.norm(P - S)
        worst = max(worst, abs(brute_force_length(bone, S, d) - info["l_bone"][j, k]))
    check(f"pose {pose}: fast == brute force (sorted pairing), 300 rays", worst < 1e-9, f"worst difference {worst:.2e} mm")

    # 5. ball chord at the projected STL centre; 6. pixel value
    uv = project_to_detector(centers_CK, pose)
    for name, (u, v) in zip(config.MARKER_NAMES, uv):
        k, j = int(np.floor((u + 100) / PIXEL_MM)), int(np.floor((v + 100) / PIXEL_MM))
        chord = info["l_balls"][j, k]
        check(f"pose {pose}: {name} ball chord at pixel (k={k}, j={j})", 2.85 <= chord <= 3.0, f"{chord:.4f} mm (predicted 2.85 - 3.0)")
        expect = (config.BONE_DENSITY_g_cm3 * info["l_bone"][j, k] + config.STEEL_DENSITY_g_cm3 * chord) / 10
        check(f"pose {pose}: {name} pixel value formula", abs(img[j, k] - expect) < 1e-12, f"sum d x = {img[j, k]:.6f} g/cm^2")

report.append(f"\nNumber of failed checks: {n_failed}  ->  "
              f"{'ALL CHECKS PASSED' if n_failed == 0 else 'SOME CHECKS FAILED, see FAIL above'}")
text = "\n".join(report)
print(text)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
with open(os.path.join(config.OUTPUT_DIR, "task5_vertebra_tests.txt"), "w") as f:
    f.write(text + "\n")
