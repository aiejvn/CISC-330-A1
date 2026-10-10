"""
compare_visual_readings.py - Task 6 visual localization: compares the marker centers read by eye from the
gridded crops (tools/export_visual_crops.py, outputs/visual/*.png) with the algorithm's result and with the
true projection. Readings are (i1, i2) in image pixels, read to about 1 px from the disc edges on the grid.

Writes outputs/task6_visual_readings.csv and outputs/task6_visual_comparison.txt.
Run from the A1 folder:   python tools/compare_visual_readings.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

READ = {   # read by eye from outputs/visual/crops_<phase>_<pose>.png, left to right
    ("sphere", "A"):   [(884, 315), (1040, 178), (1213, 877)],
    ("sphere", "B"):   [(885, 1628), (1039, 606), (1220, 1660)],
    ("vertebra", "A"): [(885, 310), (1040, 181), (1217, 883)],
    ("vertebra", "B"): [(886, 1625), (1038, 601), (1222, 1665)],
}
TRUTH = {"sphere": "markers_projected_task4.npz", "vertebra": "markers_stl_projected.npz"}
found = np.load(os.path.join(config.OUTPUT_DIR, "markers_localized_task6.npz"))
# readings recorded with tools/visual_picker.py replace the built-in ones (same phase and pose)
for (phase, pose) in list(READ):
    f = os.path.join(config.OUTPUT_DIR, "visual", f"picks_{phase}_{pose}.csv")
    if os.path.exists(f):
        pk = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)
        if len(pk) == 3:
            READ[(phase, pose)] = [tuple(r) for r in pk]
            print("using picks from", f)
rows, lines, worst = ["phase,pose,visual_i1,visual_i2,algo_i1,algo_i2,true_i1,true_i2,"
                      "visual_vs_true_px,algo_vs_true_px,visual_vs_algo_px"], [], 0.0
for (phase, pose), vis in READ.items():
    algo = found[f"{phase}_{pose}_px"]
    true = np.load(os.path.join(config.OUTPUT_DIR, TRUTH[phase]))[f"Img{pose}_px"]
    for v in vis:
        v = np.array(v, float)
        a = algo[np.argmin(np.linalg.norm(algo - v, axis=1))]
        t = true[np.argmin(np.linalg.norm(true - v, axis=1))]
        d_vt, d_at, d_va = np.linalg.norm(v - t), np.linalg.norm(a - t), np.linalg.norm(v - a)
        worst = max(worst, d_vt)
        rows.append(f"{phase},{pose},{v[0]:.1f},{v[1]:.1f},{a[0]:.2f},{a[1]:.2f},{t[0]:.2f},{t[1]:.2f},"
                    f"{d_vt:.2f},{d_at:.2f},{d_va:.2f}")
        lines.append(f"{phase:8s} {pose}: visual ({v[0]:.1f},{v[1]:.1f})  algorithm ({a[0]:.2f},{a[1]:.2f})  "
                     f"true ({t[0]:.2f},{t[1]:.2f})  visual-true {d_vt:.2f} px ({d_vt*config.PIXEL_SIZE_mm:.2f} mm)  "
                     f"algorithm-true {d_at:.2f} px ({d_at*config.PIXEL_SIZE_mm:.3f} mm)")
lines.append(f"\nlargest visual-vs-true distance: {worst:.2f} px = {worst*config.PIXEL_SIZE_mm:.2f} mm")
open(os.path.join(config.OUTPUT_DIR, "task6_visual_readings.csv"), "w").write("\n".join(rows) + "\n")
text = "\n".join(lines); print(text)
open(os.path.join(config.OUTPUT_DIR, "task6_visual_comparison.txt"), "w").write(text + "\n")
