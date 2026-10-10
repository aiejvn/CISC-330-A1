"""
config.py - shared constants for CISC/CMPE 330 HW1 (Cyber Knife navigation).

Everything that describes the whole system lives here, in one place, so every
module uses the same numbers. Units: millimetres (mm) and degrees (deg).

Coordinate frames (names and definitions from the assignment and slides)
------------------------------------------------------------------------
MD   : Model frame of the vertebra model. The STL file and the marker table
       ("Marker coordinates in Model frame") are given in it.
CK   : Cyber Knife frame, our home frame. Its center is where the two beam
       central lines intersect ("center of the CK coordinate system").
       +x points to the supine patient's right hand, +y points to the ceiling
       (anterior), +z points from toe to head (superior).
       The assignment figures look at the scene from the feet of the patient,
       so +x (the patient's right) appears on OUR left.
DetA, DetB : Detector frames, one per pose. Center = center of the detector.
       Axes (u, v, w): the detector's +u and the CK's +z are always identical;
       the detector's +w always points to the X-ray source; v completes the
       right-handed frame (u x v = w). At the hypothetical zero-angle home
       position, the detector's +v and +w equal CK's +x and +y.
ImgA, ImgB : Image frames, in pixels (resolution 0.1 mm). As in the Discrete
       Tomography slides ("Physical space & Digital modeling space"): the image
       frame's base vectors are aligned with the detector frame's, and its
       center Ci is the LOWER-LEFT corner of the image. The detector center is
       the center of the image. Only points ON the detector (w = 0) are
       meaningful in the image frame.

AI use: written with help from Claude (Anthropic), Sep 2026; reviewed and
tested by the group.
"""
import os
import numpy as np

# ---- X-ray system (assignment p.2-3) ---------------------------------------
SDD_mm = 2000.0                 # source-detector distance (200 cm)
SAD_mm = 1000.0                 # source-axis distance (100 cm): source -> CK center
POSE_ANGLE_deg = {"A": +45.0,   # pose A: source/detector pair rotated +45 deg about +z
                  "B": -45.0}   # pose B: rotated -45 deg about +z

# ---- Detectors (identical) --------------------------------------------------
DETECTOR_SIZE_mm = 200.0        # each detector is 20 cm x 20 cm square
PIXEL_SIZE_mm = 0.1             # resolution 0.1 mm
N_PIXELS = int(round(DETECTOR_SIZE_mm / PIXEL_SIZE_mm))   # 2000 pixels per side

# ---- Vertebra model and markers (assignment p.3-4) --------------------------
MARKER_NAMES = ["M1", "M2", "M3"]
MARKERS_MD_mm = np.array([[-19.0, 28.0, 11.0],    # M1 tip of right superior transverse process
                          [ 43.0, 14.0,  2.0],    # M2 tip of left transverse process
                          [  2.0, 48.0, -6.0]])   # M3 tip of spinous (posterior) process
MARKER_DIAMETER_mm = 3.0

# ---- DRR (Task 5) -------------------------------------------------------------
# Sphere phase: a sphere of radius 50 mm at the CK center, density 0.25 g/cm^3,
# with the three markers inside. The steel density and the attenuation constant
# k are NOT given in the assignment: they are our own assumptions (stated in the report).
SPHERE_RADIUS_mm = 50.0
SPHERE_DENSITY_g_cm3 = 0.25
STEEL_DENSITY_g_cm3 = 7.85      # assumed (typical steel)
ATTENUATION_K_cm2_g = 1.0       # assumed; I = I0 exp(-k sum(d x)), d in cm, x in g/cm^3
BONE_DENSITY_g_cm3 = 0.25       # vertebra model density from the assignment (same as the sphere)

# ---- How the vertebra model sits in the CK frame -------------------------------
# The centers of the MD and CK frames coincide and their base vectors are
# parallel (assignment). The vertebra lies in supine pose: in the MD frame the
# spinous process (M3) points to +y and the right side is -x, so in the CK frame
# x and y point the opposite way: (x, y, z)_CK = (-x, -y, z)_MD, a rotation of
# 180 deg about +z. "identity" is kept only for comparison.
MD_TO_CK_MODE = "rotate180"     # "rotate180" or "identity"

# ---- Files --------------------------------------------------------------------
A1_DIR = os.path.dirname(os.path.abspath(__file__))
STL_PATH = os.path.join(A1_DIR, "data", "LumbarVertebrae.-witrh-Markers.stl")
OUTPUT_DIR = os.path.join(A1_DIR, "outputs")
