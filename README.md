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
| 5 | Digitally Reconstructed Radiograph (DRR) | done, tested (sphere phase: 3 mm balls at the table positions; vertebra phase: the STL's own 2 mm balls) | `drr.py`, `run_task5_drr.py`, `tests/test_task5_drr.py`, `tests/test_task5_vertebra.py` |
| 6 | Marker Localization | done, tested (sphere and vertebra DRRs, both poses) | `localize.py`, `run_task6_localize.py`, `tests/test_task6_localization.py` |
| 7 | Marker Reconstruction + REM | done, tested | `reconstruct.py`, `tests/test_task7_reconstruction.py` |
| 8 | Marker Correspondences | done, tested (2 methods: epipolar tilt and REM) | `correspondence.py`, `tests/test_task8_correspondence.py` |
| 9 | Marker Ambiguity | **to do** | new file, e.g. `ambiguity.py` |
| 10 | Marker Reconstruction Error Analysis | **to do** | new file, e.g. `error_analysis.py` |

Tasks 9 and 10 do not need the DRR images. They start from the marker
projections that Task 4 already saved. Tasks 5, 6 and 8 are finished (details in sections 5 and 6).

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
python3 run_task5_drr.py            # sphere DRRs, then vertebra DRRs (needs data/ STL), 2000 x 2000, slow
python3 tests/test_task5_drr.py
python3 tests/test_task5_vertebra.py
python3 run_task6_localize.py      # reads the four DRR .npy files from Task 5
python3 tests/test_task6_localization.py
python3 tests/test_task8_correspondence.py
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
returns the distinct corner points. `split_vertebra_and_markers(triangles,
marker_centers_MD)` separates the bone triangles from the three 2 mm marker
balls (connected pieces of the surface) and returns the bone, the balls and
the ball centers. No extra library needed.

### `drr.py` – Task 5 (DRR)
One ray per pixel, from the source S through the middle of the pixel; image
`img[j, k]` holds the pixel value Σ d_i x_i (g/cm²; d in cm, x in g/cm³) with k = i1 (u direction)
and j = i2 (v direction), so plot with `origin="lower"`. No intensity or attenuation constant is
needed. Assumed densities (not given in the assignment, set in `config.py`): steel 7.85 g/cm³,
bone 1.0 g/cm³. All markers are 2 mm (`drr.MARKER_DIAMETER_mm`).
| function | returns | how |
|---|---|---|
| `n_pixels(pixel_mm)` | int | 2000 for 0.1 mm, 200 for 1 mm (coarse test) |
| `pixel_centers_CK(pose, pixel_mm, rows)` | (rows, n, 3) | pixel middles (u, v, 0) moved to CK with `F_CK_from_Det` |
| `render_sphere_drr(pose, pixel_mm, marker_centers_CK)` | Σ d_i x_i image | sphere phase: sphere R = 50 mm, density 0.25, with three 2 mm steel balls that *replace* sphere material along the ray (`geometry.line_sphere_chord`) |
| `mesh_path_length(triangles_CK, pose, pixel_mm)` | (length, balance) | vertebra phase: Moeller-Trumbore ray-triangle test. An exit face adds t, an entry face subtracts t, so no sorting of hits is needed. Each triangle is only tested on the pixels in the bounding box of its projection. `balance` (entries minus exits) must be 0 for a closed surface. |
| `render_vertebra_drr(pose, pixel_mm, bone, balls)` | (Σ d_i x_i, info) | bone path × bone density + ball path × steel density |

### `localize.py` – Task 6 (Marker Localization)
Works on the DRR value Σ d_i x_i itself, where a steel ball is a compact bump (about 1.5).
| function | returns | how |
|---|---|---|
| `blob_score(a)` | score image | mean of a 21 × 21 px box minus the same box mean 35 px to the left, right, up and down; the score is the smallest of the four. A bone edge is high on one side only (score ≈ 0), a ball on all four. |
| `find_peaks(score, n)` | [(k, j, score)] | n best maxima at least 8 mm apart |
| `refine_centroid(a, k, j)` | (i1, i2) px | plane fitted to the border of a 61 × 61 window, subtracted, then intensity-weighted center of the pixels above half the peak excess |
| `refine_area_centroid(a, k, j)` | (i1, i2) px | same background removal, unweighted center of the connected region above half the peak (shape only) |
| `localize_markers(img)` | dict `peak_px`, `centroid_px`, `area_px`, `score`, `next_score` | the three methods side by side; `next_score` is the 4th-best candidate and must be far below the markers' scores |
| `image_px_to_detector_mm(px)` | (u, v) mm | inverse of `detector_to_image` |

The markers are identical, so the returned order (best score first) says
nothing about M1, M2, M3. `centroid_px` is the result later tasks use.

### `correspondence.py` – Task 8 (Marker Correspondences)
A pairing is a permutation `perm`: point i on A goes with point `perm[i]` on B (N! in total, 6 for 3 markers).
| function | returns | how |
|---|---|---|
| `plane_tilt_deg(uv, pose)` | degrees | tilt of the plane through the baseline S_A S_B and the ray of a point; the same marker has the same tilt on both detectors |
| `match_by_epipolar(uv_A, uv_B)` | dict `perm`, `cost`, `tilt_A`, `tilt_B` | method 1: cost of a pairing = sum of the 3 tilt differences (degrees); smallest wins. No 3-D reconstruction, no marker table. |
| `match_by_rem(uv_A, uv_B)` | dict `perm`, `cost` | method 2: reconstruct all three pairs; cost = sum of the 3 REMs (mm); smallest wins |
| `margin(cost)` | float | second-best / best cost; near 1 means ambiguous (the Task 9 situation) |
| `correspond(uv_A, uv_B)` | dict `epipolar`, `rem`, `agree`, `perm`, `markers_CK`, `REM` | runs both methods, reports whether they `agree`, and returns the markers (CK) and REM of the method 1 pairing. The markers are identical, so the points are not named M1, M2, M3. |

### Runners and tests
* `run_task2_markers.py`: saves the markers (above) and draws
  `outputs/task2_markers_in_CK.png` (setup and markers seen from the feet).
* `run_task4_marker_projection.py`: projects the markers, saves
  `outputs/markers_projected_task4.npz` and `.csv`, prints the hand calculation,
  and draws `outputs/task4_markers_on_detectors.png`.
* `tests/test_task1_frames.py`: two points per transform with predicted results, plus each inverse.
* `tests/test_task3_projector.py`: three points projected onto both detectors with predicted results.
* `tests/test_task7_reconstruction.py`: feeds the Task 4 projections back in and checks that the true markers come out with MRE and REM near zero.
* `run_task5_drr.py`: runs the sphere phase first, then the vertebra phase. Sphere DRRs for both poses (2000 × 2000) → `outputs/drr_sphere_{A,B}.npy`, `task5_drr_sphere.png`. Vertebra phase: splits the STL into bone and its own 2 mm balls, renders both poses → `outputs/drr_vertebra_{A,B}.npy`, `task5_drr_vertebra.png`, and `markers_stl_projected.npz` (true projections of the STL ball centers, the ground truth for Task 6 on the vertebra).
* `tests/test_task5_drr.py`: chord lengths, pixel middles, sphere-only pixels against hand-computed intensities, marker contrast ≈ 0.10 at each ball (predicted 0.09–0.13), images equal away from the markers.
* `tests/test_task5_vertebra.py`: the STL splits into 3 balls and a bone with positive volume, ball volumes ≈ 4.15 mm³, ball centers within 0.5 mm of the table, entry/exit balance 0 for every pixel, fast ray test equals a brute-force sorted pairing on 300 rays, ball chords ≈ 2 mm with matching intensities.
* `run_task6_localize.py`: localizes the markers in all four DRRs → `outputs/markers_localized_task6.npz` (keys `<sphere|vertebra>_<A|B>_px` and `_uv_mm`), `task6_localization.png`.
* `tests/test_task6_localization.py`: every found marker matches one true marker one-to-one, the error is below tolerance for all three methods, and the 4th candidate is weak.
* `tests/test_task8_correspondence.py`: true order, all six orders of B, and the Task 6 localized points of the sphere and vertebra DRRs.

### `outputs/` – saved results you can reuse
| file | contents |
|---|---|
| `markers_task2.npz` | keys `MD`, `CK`, `DetA`, `DetB`: the three markers, shape (3, 3), mm |
| `markers_projected_task4.npz` | keys `DetA_uv_mm`, `DetB_uv_mm` (shape (3, 2), mm) and `ImgA_px`, `ImgB_px` (shape (3, 2), pixels): the true marker projections, rows in the order M1, M2, M3 |
| `drr_sphere_{A,B}.npy`, `drr_vertebra_{A,B}.npy` | the four DRRs, float Σ d_i x_i (g/cm²), shape (2000, 2000), indexed `[j, k]` |
| `markers_stl_projected.npz` | keys `CK` (3, 3) STL ball centers, `DetA_uv_mm`, `DetB_uv_mm`, `ImgA_px`, `ImgB_px` (3, 2): true projections of the vertebra DRR's own balls (they differ from the table-based Task 4 projections by 0.35–0.72 mm) |
| `markers_localized_task6.npz` | keys `<phase>_<pose>_px` and `<phase>_<pose>_uv_mm`, shape (3, 2), unordered |
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

## 6. Results of Tasks 5, 6, 8 and notes for the tasks still to do

**Task 5 (DRR) – done**
* Sphere phase: R = 50 mm sphere at the CK center, density 0.25 g/cm³, with the
  three 3 mm steel balls at the table positions (balls replace sphere material).
  Tests compare sphere-only pixels with hand-computed intensities and check that
  a ball darkens its pixel by a factor of about 0.10.
* Vertebra phase: the STL is split into the bone and its own three 2 mm balls
  (`split_vertebra_and_markers`). Rays are tested against the triangles with our own
  numpy Moeller-Trumbore code (no external mesh library). Every pixel has
  entry/exit balance 0 and the fast code agrees with a brute-force sorted version.
* Densities and k are our assumptions (steel 7.85, bone 1.0, k = 1.0); see `config.py`.
* The STL balls are 2 mm across and 0.34–0.44 mm from the table positions, so
  the vertebra DRR has its own ground truth, `outputs/markers_stl_projected.npz`
  (0.35–0.72 mm from the table-based Task 4 projections). Task 6 on the vertebra
  is compared against that file.

**Task 6 (Marker Localization) – done**
* Three methods are compared: `peak_px` (pixel accurate), `centroid_px`
  (intensity-weighted) and `area_px` (unweighted shape). The detector coordinates
  come from `F_Det_from_Img()`.
* Errors against the true projections (all checks in `outputs/task6_localization_tests.txt`):
  sphere centroid below 0.01 mm; vertebra centroid 0.02–0.07 mm (about 0.2–0.7 px);
  peak method up to 0.28 mm. The 4th-best candidate scores far below the weakest
  marker in all four images.
* The vertebra is harder than the sphere because the bone behind a ball makes the
  background uneven; the plane fit and the four-sided score handle this.

**Task 8 (Marker Correspondences) – done**
* Method 1 (epipolar lines: the same marker has the same tilt of the plane through
  both sources) and method 2 (smallest sum of REM) both find the true pairing for
  all six orderings of the points on detector B, and for equidistant and for four
  markers. They only find the pairing; the markers are not named.
* For the true pairing REM is ≈ 0. The best wrong pairing costs 1.23 degrees in
  summed tilt difference and 15.83 mm in summed REM, far above the correct one.
* With the localized points of Task 6 both methods still agree: sphere REM below
  0.005 mm, vertebra REM below 0.005 mm, and every marker is reconstructed within
  0.04 mm of the true position. The localized points are unordered, so this uses
  the real pipeline Task 6 → Task 8 → Task 7.

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
  Our vertebra DRR uses the STL balls as they are; the sphere DRR uses 3 mm balls at the table positions.
* Steel density, bone density and the attenuation constant k are not given; we assume 7.85 g/cm³, 1.0 g/cm³ and 1.0 cm²/g.

## Use of AI
Code written with help from Claude (Anthropic), Sep–Oct 2026, then reviewed and
tested by the group.
