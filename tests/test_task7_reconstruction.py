"""
Task 7 test: Marker Reconstruction.

Assignment: "Feed into your module the known marker locations in Detector frame
and show that you produce the known marker locations in CK frame, with zero
(or rather near zero) residual error."
The known locations on the detectors come from Task 4, the known CK positions
from Task 2.

Run from the A1 folder (Mac Terminal):   python3 tests/test_task7_reconstruction.py

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from reconstruct import reconstruct_markers
from markers import load_markers

TOLERANCE_mm = 1e-9
true_M_CK = load_markers()["CK"]                                                    # Task 2
projected = np.load(os.path.join(config.OUTPUT_DIR, "markers_projected_task4.npz"))  # Task 4
M_CK, REM_mm = reconstruct_markers(projected["DetA_uv_mm"], projected["DetB_uv_mm"])
MRE_mm = np.linalg.norm(M_CK - true_M_CK, axis=1)     # distance between reconstructed and true marker

np.set_printoptions(precision=4, suppress=True)
report_lines, n_failed = [], 0
report_lines.append("marker | detector A (u,v) mm   | detector B (u,v) mm   | reconstructed in CK  | true in CK          | MRE mm   | REM mm")
for i, name in enumerate(config.MARKER_NAMES):
    passed = MRE_mm[i] < TOLERANCE_mm and REM_mm[i] < TOLERANCE_mm
    n_failed += (not passed)
    report_lines.append(f"{name:6s} | {str(projected['DetA_uv_mm'][i]):21s} | {str(projected['DetB_uv_mm'][i]):21s} | "
                        f"{str(M_CK[i]):20s} | {str(true_M_CK[i]):19s} | {MRE_mm[i]:.1e}  | {REM_mm[i]:.1e}  {'PASS' if passed else 'FAIL'}")

report_lines.append(f"\nNumber of failed checks: {n_failed}  ->  "
                    f"{'ALL CHECKS PASSED' if n_failed == 0 else 'SOME CHECKS FAILED, see FAIL above'}")
report = "\n".join(report_lines)
print(report)
with open(os.path.join(config.OUTPUT_DIR, "task7_reconstruction_tests.txt"), "w") as f:
    f.write(report + "\n")
