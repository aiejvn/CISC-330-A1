# CISC/CMPE 330 Assignment 1 – Cyber Knife: code guide

This folder has our Python code for the Cyber Knife X-ray navigation assignment.
Read this file first. It says what each file does, how it does it, how to run
it, and what is left to do.

## 1. Status

| Task | What | Status | Where |
|---|---|---|---|
| 1 | Frame Transforms | done, tested | `frames.py`, `tests/test_task1_frames.py` |
| 2 | Marker Transforms | done | `markers.py`, `run_task2_markers.py` |
| 3 | Forward Projector | done, tested | `projector.py`, `tests/test_task3_projector.py` |
| 4 | Marker Projector | done | `run_task4_marker_projection.py` |
| 5 | Digitally Reconstructed Radiograph (DRR) | **to do** | new file, e.g. `drr.py` |
| 6 | Marker Localization | **to do** (needs the Task 5 DRRs) | new file, e.g. `localize.py` |
| 7 | Marker Reconstruction + REM | done, tested | `reconstruct.py`, `tests/test_task7_reconstruction.py` |
| 8 | Marker Correspondences | **to do** | new file, e.g. `correspondence.py` |
| 9 | Marker Ambiguity | **to do** | new file, e.g. `ambiguity.py` |
| 10 | Marker Reconstruction Error Analysis | **to do** | new file, e.g. `error_analysis.py` |

Tasks 8, 9 and 10 do not need the DRR images. They start from the marker
projections that Task 4 already saved, so they can be done in parallel with 5 and 6.

## 2. How to run

You need Python 3 with `numpy` and `matplotlib`:
```
pip install numpy matplotlib
```
Run from inside this `A1` folder, **in this order** (each step reads what the step before saved):
```
python3 tests/test_task1_frames.py
python3 run_task2_markers.py
python3 tests/test_task3_projector.py
python3 run_task4_marker_projection.py
python3 tests/test_task7_reconstruction.py
```
On Windows, type `py` or `python` instead of `python3`, and `\` instead of `/`.
No input is typed while running; all test points are written inside the test files.
Each test prints PASS/FAIL per check and ends with
`Number of failed checks: 0 -> ALL CHECKS PASSED`.

## 3. The big picture

```
         F_CK_from_MD              F_Det_from_CK
MD frame ------------> CK frame -------------> detector frame A or B   (3D coordinates only)

CK point --project_to_detector--> (u, v) on detector, mm --F_Img_from_Det--> (i1, i2) in image, pixels

(u, v) on A  +  (u, v) on B --reconstruct_point--> marker M in CK frame  +  REM
```
* A **frame transform** (4x4 matrix) only changes the coordinates of a 3D point.
* A **projection** finds where the line from the X-ray source through the point
  hits the detector plane. This is a different operation, done in `projector.py`.
* A **reconstruction** goes back: from the two detector points, draw the two
  back-projection lines to the sources and find where they (almost) meet.

## 4. Frames and conventions

* **Units:** mm and degrees; image coordinates in pixels. Variable names end in
  `_mm`, `_px`, `_deg` where it helps.
* **Frames:**
  * `MD`: the frame of the vertebra model. The STL file and the marker table are given in it.
  * `CK`: our home frame. +x = patient's right hand, +y = ceiling (anterior),
    +z = toe to head. Its center is where the two beam central lines meet.
  * `DetA`, `DetB` (the assignment's F_A, F_B): center at the detector center C.
    +u = CK +z always, +w points to the X-ray source, v completes the frame.
    At the zero-angle home position v = CK +x and w = CK +y; pose A rotates
    everything +45 deg about +z, pose B -45 deg.
  * `ImgA`, `ImgB`: pixels of 0.1 mm, 2000 x 2000. The image frame center is the
    lower left corner of the image; the detector center is pixel (1000, 1000).
    Pixel (k, j) covers i1 in [k, k+1) and i2 in [j, j+1), so its middle is
    (k + 0.5, j + 0.5).
* **Matrix names:** `F_CK_from_MD` is the course's F(CK <- MD): it takes a point
  written in the MD frame and gives the same point written in the CK frame. The
  inverse swaps the names (`F_MD_from_CK`). Matrices chain right to left.
* **Points in code:** one point has shape `(3,)`, many points have shape `(N, 3)`
  (one point per row). `geometry.apply_transform(F, points)` handles both.
* **MD -> CK is a rotation of 180 deg about z**, (x, y, z)_CK = (-x, -y, z)_MD.
  The MD frame's +x points to the patient's left and +y to the back, while the
  CK frame's +x points to the patient's right and +y to the front.
  See `docs/md_to_ck_rotation_explained.png`. The switch is `MD_TO_CK_MODE` in
  `config.py` ("identity" is kept only for comparison).
* **Letters used:** `S` X-ray source, `C` detector center, `u`, `v`, `w` detector
  axes, `Td` a point on the detector, `P1, v1, P2, v2, L1, L2, M` as in the Math
  Primer symbolic intersection.

## 5. What each file does, and how

### `config.py` – all constants in one place
SDD = 2000 mm, SAD = 1000 mm, pose angles (A +45, B -45), detector size 200 mm,
pixel size 0.1 mm, number of pixels, marker names and the marker table (MD
frame), marker diameter 3 mm, `MD_TO_CK_MODE`, and the file paths (`STL_PATH`,
`OUTPUT_DIR`). Change a number here and every file uses the new value.

### `geometry.py` – reusable Math Primer tools (no Cyber Knife specifics)
| function | returns | how |
|---|---|---|
| `normalize(v)` | unit vector | v / \|v\| |
| `translation(d)`, `rotation_x/y/z(angle_deg)`, `scaling(factor)` | 4x4 matrix | standard homogeneous matrices |
| `frame_to_home(O_e, e1, e2, e3)` | 4x4 matrix | "to home, base vectors in columns", center in the last column |
| `invert_rigid(F)` | 4x4 matrix | [R^T, -R^T t] |
| `apply_transform(F, points)` | points | adds the 4th coordinate 1, multiplies, drops it |
| `is_rigid(F)` | True/False | R^T R = I and det R = +1 |
| `line_plane_intersection(P, v, A, n)` | (points, t) | t = ((A - P).n) / (v.n); one line or many |
| `symbolic_intersection(P1, v1, P2, v2)` | (M, d, L1, L2) | solve [-v1 v2 v3](t1,t2,t3)^T = P1 - P2 with v3 = normalize(v1 x v2); M = (L1+L2)/2, d = \|L1 - L2\| |

### `frames.py` – Task 1, every frame transform
| function | returns | how |
|---|---|---|
| `F_CK_from_MD(mode=None)`, `F_MD_from_CK()` | 4x4 | rotation of 180 deg about z (or identity) |
| `detector_geometry(pose)` | dict `S_CK`, `C_CK`, `u`, `v`, `w` | home position S0 = (0, SAD, 0), C0 = (0, -(SDD-SAD), 0), rotated by the pose angle about z |
| `F_CK_from_Det(pose)`, `F_Det_from_CK(pose)` | 4x4 | `frame_to_home(C, u, v, w)` and its inverse |
| `F_Img_from_Det()`, `F_Det_from_Img()` | 4x4 | scale by 10 px/mm, then shift by (1000, 1000, 0); and the inverse |
| `all_transforms()` | dict, keys like `"DetA<-CK"` | all of the above |
| `transform_points(points, [F1, F2, ...])` | points | applies the list in order |

### `markers.py` – Task 2
`markers_in_all_frames()` gives the three markers in the MD, CK, DetA and DetB
frames. `save_markers()` writes them to `outputs/markers_task2.npz` and `.csv`;
`load_markers()` reads them back as a dict `{"MD", "CK", "DetA", "DetB"}`, each
an array of shape (3 markers, 3 coordinates).

### `projector.py` – Task 3 (Forward Projector)
| function | returns | how |
|---|---|---|
| `project_to_detector(points_CK, pose)` | (u, v) mm, shape (2,) or (N, 2) | line from S through the point, intersected with the plane through C with normal w; result moved into the detector frame (w = 0). Refuses a point at or behind the source. |
| `detector_to_image(uv_mm)` | (i1, i2) px | `F_Img_from_Det` |
| `project_to_image(points_CK, pose)` | (i1, i2) px | the two above together |
| `on_detector(uv_mm)` | True/False | \|u\|, \|v\| <= 100 mm |

### `reconstruct.py` – Task 7 (Marker Reconstruction)
| function | returns | how |
|---|---|---|
| `back_projection_line(uv_mm, pose)` | (Td_CK, unit direction to the source) | Td = F_CK_from_Det (u, v, 0); direction = (S - Td)/\|S - Td\| |
| `reconstruct_point(uv_A_mm, uv_B_mm)` | (M_CK, REM_mm, lines) | line 1 from detector A, line 2 from detector B, then `symbolic_intersection`; REM = shortest distance between the two back-projection lines |
| `reconstruct_markers(list_A, list_B)` | (M_CK (N,3), REM (N,)) | the i-th point on A is paired with the i-th point on B. **Both lists must have the same length and the same marker order.** Finding that order is Task 8. |

### `stl_io.py`
`read_binary_stl(path)` returns the triangles of the STL surface model, shape
(N, 3 corners, 3 coordinates), in mm, MD frame. `corner_points(triangles)`
returns the distinct corner points. No extra library needed.

### Runners and tests
* `run_task2_markers.py`: saves the markers (above) and draws
  `outputs/task2_markers_in_CK.png` (setup and markers seen from the feet).
* `run_task4_marker_projection.py`: projects the markers, saves
  `outputs/markers_projected_task4.npz` and `.csv`, prints the hand calculation,
  and draws `outputs/task4_markers_on_detectors.png`.
* `tests/test_task1_frames.py`: two points per transform with predicted results, plus each inverse.
* `tests/test_task3_projector.py`: three points projected onto both detectors with predicted results.
* `tests/test_task7_reconstruction.py`: feeds the Task 4 projections back in and checks that the true markers come out with MRE and REM near zero.

### `outputs/` – saved results you can reuse
| file | contents |
|---|---|
| `markers_task2.npz` | keys `MD`, `CK`, `DetA`, `DetB`: the three markers, shape (3, 3), mm |
| `markers_projected_task4.npz` | keys `DetA_uv_mm`, `DetB_uv_mm` (shape (3, 2), mm) and `ImgA_px`, `ImgB_px` (shape (3, 2), pixels): the true marker projections, rows in the order M1, M2, M3 |
| `*.csv` | the same numbers, easy to open |
| `*_tests.txt`, `task*_*.txt` | what each script printed |

Load them like this:
```python
import numpy as np
from markers import load_markers
true_M_CK = load_markers()["CK"]                                   # (3, 3)
proj = np.load("outputs/markers_projected_task4.npz")
uv_A, uv_B = proj["DetA_uv_mm"], proj["DetB_uv_mm"]               # (3, 2) each
```

### `data/`, `tools/`, `docs/`
* `data/LumbarVertebrae.-witrh-Markers.stl`: the vertebra surface model (MD frame).
* `tools/check_stl_markers.py`: measures the three marker balls inside the STL
  (result in `outputs/stl_marker_check.txt`). Not part of the assignment; it
  matters for Task 5 (see below).
* `tools/show_point_in_all_frames.py`: edit the `POINTS` list, run, and it
  prints any point in every frame and where it projects. Handy for checking by hand.
* `docs/md_to_ck_rotation_explained.png`: why MD -> CK is a 180 deg rotation.

## 6. Notes for the tasks still to do

**Task 5 (DRR)**
* Geometry is ready: `detector_geometry(pose)` gives S, C, u, v, w. The middle
  of pixel (k, j) in the CK frame is
  `apply_transform(F_CK_from_Det(pose), apply_transform(F_Det_from_Img(), [k+0.5, j+0.5, 0]))`.
  Every pixel's line goes from S to that point.
* Sphere phase: radius R = 50 mm, centered at the CK center, density
  0.25 g/cm³, with the three 3 mm steel markers inside. We need a line-sphere
  intersection (Math Primer) to get the length of each line inside a sphere;
  it is not in `geometry.py` yet.
* Attenuation from the slides: I = I0 · exp(-k · Σ d_i x_i), with d the length
  inside each object and x its density. The steel density is not given, so we
  must state the value we assume. The assignment lets us choose the intensity range.
* Vertebra phase: lines against the STL triangles. The assignment suggests
  using public code (for example trimesh or Open3D, with credit).
* **Open question:** the STL's marker balls are 2 mm across, not 3 mm, and
  their centers are 0.34–0.44 mm from the marker table (see
  `outputs/stl_marker_check.txt`). We have to decide which markers the DRR uses,
  because Task 6 compares against the table positions.
* Start with a coarse detector (assignment tip) and save the images to `outputs/`.

**Task 6 (Marker Localization)**: find the marker centers in the DRR images
(pixels), convert to detector mm with `F_Det_from_Img()`, and compare with the
true projections in `markers_projected_task4.npz` and with positions read off
by eye in an image viewer.

**Task 8 (Marker Correspondences)**: the markers are identical, so on each
detector we only get three points in unknown order. `reconstruct_point` gives
the REM for any pairing of one point on A with one point on B; the true pairs
should give near-zero REM. A second method is required (an idea: compare the
distances between the reconstructed markers with the known distances between
markers). Test with the true projections, then with two markers swapped on one detector.

**Task 9 (Marker Ambiguity)**: ambiguity happens when a wrong pairing also gives
near-zero REM, i.e. two markers lie in the same plane through S_A and S_B.
Both sources are in the plane z = 0, so for example two markers both at z = 0
give an ambiguity. Move markers in the CK frame, project them with
`project_to_detector`, and show the method detects it. Again two methods.

**Task 10 (Error Analysis)**: for MLE = 0, 1, 2, ... mm, shift a marker's
(u, v) on detector A by MLE in direction θ_A and on detector B in direction θ_B,
θ from 0 to 359 deg in 1 deg steps (360 x 360 = 129,600 cases per MLE per
marker). Reconstruct each case with `reconstruct_point`, then
MRE = |M - true marker| (true markers from `load_markers()["CK"]`) and REM.
maxMRE = 5.0 mm. A Python loop over all cases works but is slow; `np.linalg.solve`
also accepts a whole stack of 3x3 systems at once.

## 7. Rules we follow (course General Instructions)
* No interactive input; test inputs are written in the test files.
* Implementation files and test files are separate.
* Fixed random seed if anything is random.
* For each test: say why the points were chosen, predict the result, then run.
* Credit any code or package we did not write.

## 8. Open questions for the instructor
* MD -> CK: we use a rotation of 180 deg about z (see section 4).
* STL marker balls: 2 mm and slightly off the table positions, while the assignment says 3 mm.

## Use of AI
Code written with help from Claude (Anthropic), Sep–Oct 2026, then reviewed and
tested by the group.
