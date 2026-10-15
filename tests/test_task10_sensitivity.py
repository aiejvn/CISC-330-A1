import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. GEOMETRI OCH REKONSTRUKTION (Task 1-7)
# ==========================================

SAD = 1000.0  # mm
SDD = 2000.0  # mm

SOURCE_A = np.array([SAD * np.cos(np.deg2rad(45)),  SAD * np.sin(np.deg2rad(45)),  0.0])
SOURCE_B = np.array([SAD * np.cos(np.deg2rad(-45)), SAD * np.sin(np.deg2rad(-45)), 0.0])

def project_point_to_detector(pt_3d, pose="A"):
    """Projicerar en 3D-punkt till detektorns (u, v)-koordinater i mm."""
    S = SOURCE_A if pose == "A" else SOURCE_B
    angle = np.deg2rad(45 if pose == "A" else -45)
    det_center = np.array([-(SDD - SAD) * np.cos(angle), -(SDD - SAD) * np.sin(angle), 0.0])
    det_normal = np.array([-np.cos(angle), -np.sin(angle), 0.0])
    
    ray_dir = pt_3d - S
    t = np.dot(det_center - S, det_normal) / np.dot(ray_dir, det_normal)
    intersect_ck = S + t * ray_dir
    
    rel = intersect_ck - det_center
    v = rel[2]
    u_axis = np.array([-np.sin(angle), np.cos(angle), 0.0])
    u = np.dot(rel, u_axis)
    return np.array([u, v])

def reconstruct_single_marker(pt_a_uv, pt_b_uv):
    """Symbolic intersection enligt Task 7."""
    angle_a = np.deg2rad(45)
    det_center_a = np.array([-(SDD - SAD) * np.cos(angle_a), -(SDD - SAD) * np.sin(angle_a), 0.0])
    u_axis_a = np.array([-np.sin(angle_a), np.cos(angle_a), 0.0])
    P1 = det_center_a + pt_a_uv[0] * u_axis_a + np.array([0, 0, pt_a_uv[1]])
    
    angle_b = np.deg2rad(-45)
    det_center_b = np.array([-(SDD - SAD) * np.cos(angle_b), -(SDD - SAD) * np.sin(angle_b), 0.0])
    u_axis_b = np.array([-np.sin(angle_b), np.cos(angle_b), 0.0])
    P2 = det_center_b + pt_b_uv[0] * u_axis_b + np.array([0, 0, pt_b_uv[1]])
    
    v1 = (SOURCE_A - P1) / np.linalg.norm(SOURCE_A - P1)
    v2 = (SOURCE_B - P2) / np.linalg.norm(SOURCE_B - P2)
    v3 = np.cross(v1, v2)
    v3 = v3 / np.linalg.norm(v3)
    
    A = np.column_stack([-v1, v2, v3])
    t = np.linalg.solve(A, P1 - P2)
    
    L1 = P1 + t[0] * v1
    L2 = P2 + t[1] * v2
    
    M = (L1 + L2) / 2.0
    rem = np.linalg.norm(L1 - L2)
    return M, rem

# ==========================================
# 2. TASK 10: SENSITIVITY & ERROR ANALYSIS
# ==========================================

def run_task10_sensitivity(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanna markörer i CK-ramen
    markers_ck = np.array([
        [19.0, -28.0, 11.0],
        [-43.0, -14.0, 2.0],
        [-2.0, -48.0, -6.0]
    ])
    
    # Felfria 2D-projektioner
    pts_A_nominal = np.array([project_point_to_detector(m, "A") for m in markers_ck])
    pts_B_nominal = np.array([project_point_to_detector(m, "B") for m in markers_ck])
    
    # Svep över fel i 2D (MLE: 0.0 till 8.0 mm)
    mle_radii = np.linspace(0.0, 8.0, 81)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    
    max_mre_list = []
    mean_mre_list = []
    max_rem_list = []
    
    # Listor för punktdiagrammet (REM vs MRE)
    scatter_mre = []
    scatter_rem = []
    
    critical_mle = None
    
    for r in mle_radii:
        mre_for_r = []
        rem_for_r = []
        
        if r == 0.0:
            for idx in range(3):
                M_rec, rem = reconstruct_single_marker(pts_A_nominal[idx], pts_B_nominal[idx])
                mre = np.linalg.norm(M_rec - markers_ck[idx])
                mre_for_r.append(mre)
                rem_for_r.append(rem)
                scatter_mre.append(mre)
                scatter_rem.append(rem)
        else:
            for theta_a in angles[::4]:
                for theta_b in angles[::4]:
                    noise_a = np.array([r * np.cos(theta_a), r * np.sin(theta_a)])
                    noise_b = np.array([r * np.cos(theta_b), r * np.sin(theta_b)])
                    
                    for idx in range(3):
                        pA = pts_A_nominal[idx] + noise_a
                        pB = pts_B_nominal[idx] + noise_b
                        M_rec, rem = reconstruct_single_marker(pA, pB)
                        mre = np.linalg.norm(M_rec - markers_ck[idx])
                        mre_for_r.append(mre)
                        rem_for_r.append(rem)
                        
                        # Spara ett urval för plot 2 så inte scatter-diagrammet blir för tungt
                        if r in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
                            scatter_mre.append(mre)
                            scatter_rem.append(rem)
                        
        max_mre = np.max(mre_for_r)
        mean_mre = np.mean(mre_for_r)
        max_rem = np.max(rem_for_r)
        
        max_mre_list.append(max_mre)
        mean_mre_list.append(mean_mre)
        max_rem_list.append(max_rem)
        
        if critical_mle is None and max_mre >= 5.0:
            critical_mle = r

    # Spara tabellsammanfattning
    summary_path = os.path.join(output_dir, "task10_sensitivity_summary.txt")
    crit_str = f"~{critical_mle:.2f} mm" if critical_mle is not None else "Not reached"
    
    with open(summary_path, "w") as f:
        f.write("=== TASK 10 PERTURBATION ANALYSIS ===\n")
        f.write(f"Max allowed MLE for MRE < 5.0 mm: {crit_str}\n\n")
        f.write(f"{'MLE (mm)':<10} | {'Mean MRE (mm)':<15} | {'Max MRE (mm)':<15} | {'Max REM (mm)':<15}\n")
        f.write("-" * 60 + "\n")
        for r, m_mre, mx_mre, mx_rem in zip(mle_radii[::8], mean_mre_list[::8], max_mre_list[::8], max_rem_list[::8]):
            f.write(f"{r:<10.2f} | {m_mre:<15.4f} | {mx_mre:<15.4f} | {mx_rem:<15.4f}\n")
            
    print(f"\n[DONE] Summary saved to: {summary_path}")
    print(f"--> Critical MLE (Max allowed 2D detector error): {crit_str}\n")
    
    # ==========================================
    # 3. GENERERA DE TVÅ DELGRAFERNA
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- PLOT 1: MLE vs. MRE ---
    ax1.plot(mle_radii, max_mre_list, 'r-', label="maxMRE")
    ax1.plot(mle_radii, mean_mre_list, 'r--', label="meanMRE")
    ax1.axhline(5.0, color='black', linestyle=':', label="maxMRE = 5.0 mm (Clinical Limit)")
    if critical_mle is not None:
        ax1.axvline(critical_mle, color='green', linestyle='--', label=f"maxMLE = {critical_mle:.2f} mm")
    ax1.set_xlabel("MLE in 2D (mm)")
    ax1.set_ylabel("MRE in 3D (mm)")
    ax1.set_ylim(-0.2, 7.5)
    ax1.set_title("Relationship: MLE vs. MRE")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # --- PLOT 2: REM vs. MRE ---
    ax2.scatter(scatter_rem, scatter_mre, alpha=0.25, c='blue', s=12, label="Reconstructions")
    ax2.axhline(5.0, color='black', linestyle=':', label="maxMRE = 5.0 mm Limit")
    ax2.set_xlabel("REM (mm)")
    ax2.set_ylabel("MRE in 3D (mm)")
    ax2.set_ylim(-0.2, 7.5)
    ax2.set_title("Relationship: REM vs. MRE")
    ax2.grid(True)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "task10_error_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[DONE] Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_task10_sensitivity()