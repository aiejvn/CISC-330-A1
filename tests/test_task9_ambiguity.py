"""
Task 9 test: Marker Ambiguity.

Assignment: "Develop a module to detect ambiguous correspondences for the reconstruction
of the three identical markers localized in their Detector frames.
Alter the known marker positions in the CK frame to create an ambiguity. Explain what causes
ambiguity, how you can engineer one. Project the resulting markers from CK frame onto their
detectors and show that you can detect the ambiguity. Solve the problem with a different
computational approach and run the same tests as above."

Run from the CISC-330-A1 folder (Terminal):   python3 tests/test_task9_ambiguity.py

AI use: written with help from Gemini / Claude; reviewed and tested by the group.
"""
import os, sys
from itertools import permutations
import numpy as np

# Se till att projektets rotmapp finns i python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from reconstruct import reconstruct_markers
from markers import load_markers
from projector import project_to_detector


# ==============================================================================
# 1. HJÄLPFUNKTIONER & METODER FÖR ATT DETEKTERA TVETYDIGHET
# ==============================================================================

def compute_rem_matrix(pts_A, pts_B):
    """
    Beräknar 3x3 REM-matrisen (kortaste 3D-avståndet mellan alla strålpar Ai och Bj).
    """
    rem_mat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            _, rem = reconstruct_markers(pts_A[i:i+1], pts_B[j:j+1])
            rem_mat[i, j] = rem[0]
    return rem_mat


def method1_rem_ambiguity(rem_matrix, threshold_mm=0.5):
    """
    Metod 1: Identifierar tvetydighet via REM-matrisen.
    Om en rad eller kolumn innehåller fler än 1 kandidat nära noll
    delar strålarna samma epipolärplan och korsar varandra i flera punkter.
    """
    row_candidates = (rem_matrix < threshold_mm).sum(axis=1)
    col_candidates = (rem_matrix < threshold_mm).sum(axis=0)
    is_ambiguous = bool(np.any(row_candidates > 1) or np.any(col_candidates > 1))
    return is_ambiguous, row_candidates, col_candidates


def method2_geometric_ambiguity(pts_A, pts_B, rem_matrix, true_distances, threshold_mm=0.5):
    """
    Metod 2: Identifierar tvetydighet genom stel kroppskongruens (triangellängder).
    Undersöker alla 3! = 6 permutationer mellan A och B. Om mer än en
    1-till-1-matchning ger låg total REM har systemet en tvetydighet.
    Avstånden i de rekonstruerade 3D-trianglarna jämförs med den sanna modellen.
    """
    valid_permutations = []
    
    for p in permutations(range(3)):
        # p[i] anger vilken detektorpunkt i B som paras med A[i]
        total_rem = sum(rem_matrix[i, p[i]] for i in range(3))
        if total_rem < 3 * threshold_mm:
            b_permuted = pts_B[list(p)]
            m_rec, _ = reconstruct_markers(pts_A, b_permuted)
            
            # Beräkna de tre inbördes avstånden i markörtriangeln
            d01 = np.linalg.norm(m_rec[0] - m_rec[1])
            d12 = np.linalg.norm(m_rec[1] - m_rec[2])
            d20 = np.linalg.norm(m_rec[2] - m_rec[0])
            recon_dist = np.sort([d01, d12, d20])
            
            dist_error = np.linalg.norm(recon_dist - np.sort(true_distances))
            valid_permutations.append({
                'perm': p,
                'total_rem': total_rem,
                'dist_error': dist_error,
                'points_3d': m_rec
            })
            
    is_ambiguous = len(valid_permutations) > 1
    return is_ambiguous, valid_permutations


# ==============================================================================
# 2. TESTKÖRNING OCH RAPPORTGENERERING
# ==============================================================================

np.set_printoptions(precision=4, suppress=True)
report_lines = []
report_lines.append("=" * 78)
report_lines.append("TASK 9: MARKER AMBIGUITY DETECTION TEST REPORT")
report_lines.append("=" * 78)

# Ladda sanna markörer från Task 2 och beräkna modellens fasta avstånd
true_M_CK = load_markers()["CK"]
true_dist = [
    np.linalg.norm(true_M_CK[0] - true_M_CK[1]),
    np.linalg.norm(true_M_CK[1] - true_M_CK[2]),
    np.linalg.norm(true_M_CK[2] - true_M_CK[0])
]

# --- TESTFALL 1: Standardkonfiguration (icke-tvetydig) ---
report_lines.append("\n--- CASE 1: Standard Non-Ambiguous Markers ---")
proj_std = np.load(os.path.join(config.OUTPUT_DIR, "markers_projected_task4.npz"))
ptsA_std = proj_std["DetA_uv_mm"]
ptsB_std = proj_std["DetB_uv_mm"]

rem_std = compute_rem_matrix(ptsA_std, ptsB_std)
ambig_rem_1, _, _ = method1_rem_ambiguity(rem_std)
ambig_geo_1, _ = method2_geometric_ambiguity(ptsA_std, ptsB_std, rem_std, true_dist)

report_lines.append("REM Matrix (mm):")
report_lines.append(str(rem_std))
report_lines.append(f"Method 1 (REM matrix) detected ambiguity:     {ambig_rem_1} (Expected: False)")
report_lines.append(f"Method 2 (Rigid geometry) detected ambiguity: {ambig_geo_1} (Expected: False)")
case1_pass = (not ambig_rem_1) and (not ambig_geo_1)
report_lines.append(f"Case 1 Status: {'PASS' if case1_pass else 'FAIL'}")


# --- TESTFALL 2: Konstruerad tvetydighet (M1 och M2 delar z-plan) ---
report_lines.append("\n--- CASE 2: Engineered Ambiguity (Coplanar Epipolar Rays) ---")
# Vi skapar en tvetydighet genom att sätta M1 och M2 till exakt samma z-koordinat (z = 0.0)
M_ambig = np.copy(true_M_CK)
M_ambig[0, 2] = 0.0   # M1 z-värde
M_ambig[1, 2] = 10.0   # M2 z-värde

report_lines.append("Engineered 3D markers in CK frame (M1 and M2 share z = 0.0):")
for i, name in enumerate(config.MARKER_NAMES):
    report_lines.append(f"  {name}: {M_ambig[i]}")

# Projicera direkt med din projector-funktion
detA_ambig = project_to_detector(M_ambig, "A")
detB_ambig = project_to_detector(M_ambig, "B")

# Beräkna REM-matrisen för det tvetydiga fallet
rem_ambig = compute_rem_matrix(detA_ambig, detB_ambig)
report_lines.append("\nComputed Ambiguous REM Matrix (mm):")
report_lines.append(str(rem_ambig))

# Testa båda metoderna på det tvetydiga fallet
ambig_rem_2, rows, cols = method1_rem_ambiguity(rem_ambig, threshold_mm=0.1)
ambig_geo_2, valid_perms = method2_geometric_ambiguity(detA_ambig, detB_ambig, rem_ambig, true_dist, threshold_mm=0.1)

report_lines.append(f"\nMethod 1: Ambiguity detected: {ambig_rem_2}")
report_lines.append(f"          Near-zero candidates per row: {rows}, per column: {cols}")
report_lines.append(f"Method 2: Ambiguity detected: {ambig_geo_2}")
report_lines.append(f"          Number of low-REM permutations: {len(valid_perms)}")

for idx, p_info in enumerate(valid_perms):
    report_lines.append(f"          Candidate {idx+1}: Permutation {p_info['perm']} | "
                        f"Sum REM = {p_info['total_rem']:.4f} mm | Tri-distance error = {p_info['dist_error']:.4f} mm")

case2_pass = ambig_rem_2 and ambig_geo_2
report_lines.append(f"Case 2 Status: {'PASS' if case2_pass else 'FAIL'}")

# Slutresultat
all_passed = case1_pass and case2_pass
report_lines.append("\n" + "=" * 78)
report_lines.append(f"FINAL RESULT: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
report_lines.append("=" * 78)

report_str = "\n".join(report_lines)
print(report_str)

# Spara till outputs-katalogen
output_file = os.path.join(config.OUTPUT_DIR, "task9_ambiguity_tests.txt")
with open(output_file, "w") as f:
    f.write(report_str + "\n")
print(f"\nRapport sparad i: {output_file}")