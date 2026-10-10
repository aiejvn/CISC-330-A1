"""
Task 6 test: Marker Localization on the saved DRRs (Task 5), both phases, both poses.

Ground truth: sphere phase = the Task 4 projections of the table markers
(outputs/markers_projected_task4.npz); vertebra phase = the projections of the STL's own
ball centers (outputs/markers_stl_projected.npz).

Why these checks, and what we predict before running:
 1. Exactly 3 candidates stand out: the 4th best detection score is below 0.3 x the weakest
    marker score (measured while developing: markers 2.0 - 2.2, 4th candidate <= 0.07).
 2. Every true marker has exactly one found marker nearest to it (a one-to-one match), so no
    marker is missed and none is found twice.
 3. Sphere phase, sub-pixel centroid: the shadow of a perfect ball over a smooth background, so
    the centroid should agree with the projected center to far below a pixel. Predicted < 0.2 px
    (0.02 mm). The pixel-accurate peak is only expected within 3 px.
 4. Vertebra phase: the ball sits on a structured bone shadow, so the centroid
    is less exact. Predicted < 2 px (0.2 mm) for the centroid, < 6 px for the peak (the shadow of a 3 mm ball is 60 px wide, so the pixel-accurate peak of step 1 is only good to a few pixels).
 5. The conversion pixels -> detector mm inverts projector.detector_to_image (round trip < 1e-9).

Run from the A1 folder:   python tests/test_task6_localization.py

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from localize import localize_markers, image_px_to_detector_mm
from projector import detector_to_image

TRUTH = {"sphere": "markers_projected_task4.npz", "vertebra": "markers_stl_projected.npz"}
TOL_PX = {"sphere": {"centroid_px": 0.2, "area_px": 0.2, "peak_px": 3.0},
          "vertebra": {"centroid_px": 2.0, "area_px": 2.0, "peak_px": 6.0}}
report, n_failed = [], 0


def check(name, ok, detail=""):
    global n_failed
    n_failed += (not ok)
    report.append(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


for phase, truth_file in TRUTH.items():
    truth = np.load(os.path.join(config.OUTPUT_DIR, truth_file))
    for pose in ("A", "B"):
        res = localize_markers(np.load(os.path.join(config.OUTPUT_DIR, f"drr_{phase}_{pose}.npy")))
        true_px = truth[f"Img{pose}_px"]
        tag = f"{phase} {pose}"
        # 1. only three candidates stand out
        check(f"{tag}: 4th candidate is weak", res["next_score"] < 0.3 * res["score"].min(),
              f"(4th {res['next_score']:.3f}, weakest marker {res['score'].min():.3f})")
        for key in ("centroid_px", "area_px", "peak_px"):
            found = res[key]
            dist = np.linalg.norm(found[:, None, :] - true_px[None, :, :], axis=2)   # (found, true)
            nearest = dist.argmin(axis=1)
            # 2. one-to-one match
            check(f"{tag} {key}: one-to-one match with the true markers", sorted(nearest) == [0, 1, 2],
                  f"(found -> true: {list(map(int, nearest))})")
            # 3./4. accuracy
            err = dist.min(axis=1)
            check(f"{tag} {key}: error below {TOL_PX[phase][key]} px", np.all(err < TOL_PX[phase][key]),
                  f"(errors {np.round(err, 2)} px = {np.round(err * config.PIXEL_SIZE_mm, 3)} mm)")

# 5. pixel -> detector mm is the inverse of detector -> pixel
uv = np.array([[0.0, 0.0], [21.29, -12.32], [-99.95, 99.95]])
check("pixel -> detector mm inverts detector -> pixel",
      np.allclose(image_px_to_detector_mm(detector_to_image(uv)), uv, atol=1e-9))

report.append(f"\nNumber of failed checks: {n_failed}  ->  "
              f"{'ALL CHECKS PASSED' if n_failed == 0 else 'SOME CHECKS FAILED, see FAIL above'}")
text = "\n".join(report)
print(text)
with open(os.path.join(config.OUTPUT_DIR, "task6_localization_tests.txt"), "w") as f:
    f.write(text + "\n")
