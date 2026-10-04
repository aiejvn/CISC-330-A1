"""
Task 3 test: Forward Projector.

Assignment: test the projector on 3 suitable points, explain the choice,
predict the result, run the module and show it gives the predicted result.
Each point is projected onto both detectors.

Run from the A1 folder (Mac Terminal):   python3 tests/test_task3_projector.py

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, frames
from projector import project_to_detector, project_to_image

det_A = frames.detector_geometry("A")

# (test point in CK, predicted (u, v) on detector A, predicted (u, v) on detector B, reason)
TEST_CASES = [
    ([0, 0, 0], [0, 0], [0, 0],
     "both beam central lines go through the CK center and hit the detector centers"),
    ([0, 0, 10], [20, 0], [20, 0],
     "+z is +u; the point is level with the CK center, halfway between source and detector, so its offset doubles"),
    (10 * det_A["v"], [0, 20], [0, 0],
     "10 mm along v_A from the CK center doubles to 20 mm on A; v_A = w_B, so on B it is on the beam central line"),
]

np.set_printoptions(precision=3, suppress=True)
report_lines, n_failed = [], 0
for point_CK, predicted_A, predicted_B, reason in TEST_CASES:
    for pose, predicted in (("A", predicted_A), ("B", predicted_B)):
        uv_mm = project_to_detector(point_CK, pose)
        i1_i2_px = project_to_image(point_CK, pose)
        passed = np.abs(uv_mm - predicted).max() < 1e-9
        n_failed += (not passed)
        report_lines.append(f"point {str(np.round(point_CK, 3)):20s} detector {pose}: predicted (u,v) {str(np.array(predicted)):10s} "
                            f"computed {str(uv_mm):14s} = pixel {str(np.round(i1_i2_px, 1)):16s} {'PASS' if passed else 'FAIL'}")
    report_lines.append(f"      why: {reason}")

report_lines.append(f"\nNumber of failed checks: {n_failed}  ->  "
                    f"{'ALL CHECKS PASSED' if n_failed == 0 else 'SOME CHECKS FAILED, see FAIL above'}")
report = "\n".join(report_lines)
print(report)
with open(os.path.join(config.OUTPUT_DIR, "task3_projector_tests.txt"), "w") as f:
    f.write(report + "\n")
