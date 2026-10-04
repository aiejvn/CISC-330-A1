"""
Task 1 test: Frame Transforms.

Assignment: test each frame transform on suitable points, explain the choice,
predict the result, run the module and show it gives the predicted result.
We use two points per transformation. Each point is also sent back through the
inverse transformation, which must return the original point.

Run from the A1 folder (Mac Terminal):   python3 tests/test_task1_frames.py

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, frames
from geometry import apply_transform

TOLERANCE_mm = 1e-9
det_A, det_B = frames.detector_geometry("A"), frames.detector_geometry("B")
F = frames.all_transforms()

# (name of forward transformation, name of its inverse, test point, predicted result, reason)
TEST_CASES = [
    ("CK<-MD", "MD<-CK", [0, 0, 0], [0, 0, 0], "the centers of the MD and CK frames coincide"),
    ("CK<-MD", "MD<-CK", [2, 48, -6], [-2, -48, -6], "M3 (spinous process) points to the table in supine pose"),
    ("DetA<-CK", "CK<-DetA", det_A["S_CK"], [0, 0, 2000], "the X-ray source is on the w axis, SDD = 2000 mm from the detector center"),
    ("DetA<-CK", "CK<-DetA", 10 * det_A["v"] + 10 * det_A["u"], [10, 10, 1000],
     "10 mm along v_A and 10 mm along u from the CK center, which is SDD - SAD = 1000 mm from the detector"),
    ("DetB<-CK", "CK<-DetB", det_B["S_CK"], [0, 0, 2000], "same as for detector A"),
    ("DetB<-CK", "CK<-DetB", 10 * det_B["v"] + 10 * det_B["u"], [10, 10, 1000], "same as for detector A, with v_B"),
    ("ImgA<-DetA", "DetA<-ImgA", [0, 0, 0], [1000, 1000, 0], "the detector center is the center of the image"),
    ("ImgA<-DetA", "DetA<-ImgA", [-100, -100, 0], [0, 0, 0], "the lower left corner of the detector is the image frame center"),
    ("ImgB<-DetB", "DetB<-ImgB", [0, 0, 0], [1000, 1000, 0], "same as for detector A"),
    ("ImgB<-DetB", "DetB<-ImgB", [-100, -100, 0], [0, 0, 0], "same as for detector A"),
]

np.set_printoptions(precision=3, suppress=True)
report_lines = [f"MD->CK mode: {config.MD_TO_CK_MODE}", ""]
n_failed = 0
for forward, inverse, point, predicted, reason in TEST_CASES:
    computed = apply_transform(F[forward], point)
    back = apply_transform(F[inverse], computed)
    forward_ok = np.abs(computed - np.asarray(predicted, float)).max() < TOLERANCE_mm
    inverse_ok = np.abs(back - np.asarray(point, float)).max() < TOLERANCE_mm
    n_failed += (not forward_ok) + (not inverse_ok)
    report_lines.append(f"{forward:11s} point {str(np.round(point, 3)):26s} predicted {str(np.round(predicted, 3)):18s} "
                        f"computed {str(computed):22s} {'PASS' if forward_ok else 'FAIL'} | inverse gives the point back: "
                        f"{'PASS' if inverse_ok else 'FAIL'} | {reason}")

report_lines.append(f"\nNumber of failed checks: {n_failed}  ->  "
                    f"{'ALL CHECKS PASSED' if n_failed == 0 else 'SOME CHECKS FAILED, see FAIL above'}")
report = "\n".join(report_lines)
print(report)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
with open(os.path.join(config.OUTPUT_DIR, "task1_frame_tests.txt"), "w") as f:
    f.write(report + "\n")
