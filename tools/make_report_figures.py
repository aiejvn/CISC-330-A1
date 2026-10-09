"""
make_report_figures.py - diagrams for the report sections of Tasks 5, 6 and 8.

Reads only files that the task scripts already saved in outputs/ and writes PNGs to
cisc330_a1_g1/figures/. Run from the A1 folder:   python tools/make_report_figures.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, frames
from geometry import line_sphere_chord, apply_transform
from markers import load_markers
from reconstruct import reconstruct_point, back_projection_line
from correspondence import PERMUTATIONS, correspond
import localize

OUT = os.path.join(config.A1_DIR, "cisc330_a1_g1", "figures")
os.makedirs(OUT, exist_ok=True)
CA, CB, OK, BAD = "#e07b00", "#6a3fa0", "#2a9d3c", "#c62828"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
load = lambda n: np.load(os.path.join(config.OUTPUT_DIR, n))
M_CK = load_markers()["CK"]
SA, SB = (frames.detector_geometry(p)["S_CK"] for p in "AB")


# ---------------------------------------------------------------- Task 5, figure 1: geometry
def fig_task5_geometry():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    # (a) top view of pose A: source, sphere, rays, detector
    g = frames.detector_geometry("A")
    S, C, v = g["S_CK"][:2], g["C_CK"][:2], g["v"][:2]
    ax.add_patch(Circle((0, 0), 50, fc="#dddddd", ec="k", zorder=3))
    for m in M_CK:
        ax.add_patch(Circle(m[:2], 6, fc="k", zorder=4))
    ax.plot(*np.array([C - 100 * v, C + 100 * v]).T, color=CA, lw=5, solid_capstyle="butt")
    ax.plot(*np.array([S, C]).T, "--", color="gray", lw=1)
    for s in (-1, 0.4, 1):
        T = C + 100 * s * v
        ax.plot(*np.array([S, T]).T, color=CA, lw=2 if s == 0.4 else 1)
    ax.plot(*S, "o", color=CA, ms=9)
    ax.text(S[0] - 30, S[1] + 40, r"source $S_A$", ha="right")
    ax.text(C[0] + 60, C[1] - 60, "detector A\n(2000 pixels)", color=CA)
    ax.text(0, -75, "sphere, $R=50$ mm\n$\\rho=0.25$ g/cm$^3$", ha="center", va="top", fontsize=9)
    ax.annotate("steel balls", xy=(M_CK[0, 0], M_CK[0, 1]), xytext=(200, 120), arrowprops=dict(arrowstyle="->"))
    T = C + 40 * v
    ax.annotate("one ray per pixel,\nthrough the pixel middle", xy=(S + 0.35 * (T - S)),
                xytext=(-800, -600), color=CA, arrowprops=dict(arrowstyle="->", color=CA))
    ax.set_aspect("equal"); ax.set_xlim(-850, 900); ax.set_ylim(-850, 850)
    ax.set_xlabel("$x_{CK}$ (mm)"); ax.set_ylabel("$y_{CK}$ (mm)")
    ax.set_title("(a) one detector, seen from the feet\n($z$ points out of the page)")
    # (b) one ray through the sphere and a ball
    bx.add_patch(Circle((0, 0), 1, fc="#dddddd", ec="k"))
    bx.add_patch(Circle((0.25, 0.1), 0.18, fc="#444444", ec="k"))
    y = 0.1
    bx.annotate("", xy=(1.35, y), xytext=(-1.35, y), arrowprops=dict(arrowstyle="->", color=CA, lw=1.5))
    xs = np.sqrt(1 - y ** 2)
    bx.plot([-xs, xs], [y + 0.45] * 2, color="k", lw=1)
    bx.text(0, y + 0.5, r"$l_{\rm sphere}$", ha="center")
    bx.plot([0.25 - 0.18, 0.25 + 0.18], [y - 0.3] * 2, color="k", lw=1)
    bx.text(0.25, y - 0.34, r"$l_{\rm ball}$", ha="center", va="top")
    bx.text(-1.35, y + 0.07, "from $S$", color=CA)
    bx.text(1.35, y + 0.07, "to pixel", color=CA, ha="right")
    bx.text(0, -1.25, r"$\sum_i d_i x_i=\frac{1}{10}\,[\,0.25\,(l_{\rm sphere}-l_{\rm ball})+7.85\,l_{\rm ball}\,]$", ha="center", fontsize=10)
    bx.set_aspect("equal"); bx.set_xlim(-1.45, 1.45); bx.set_ylim(-1.7, 1.3); bx.axis("off")
    bx.set_title("(b) path of one ray (a ball replaces sphere material)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task5_geometry.png"), dpi=150); plt.close(fig)


# ---------------------------------------------------------------- Task 5, figure 2: profile + signed length
def fig_task5_profile():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    img = load("drr_sphere_A.npy")
    j = 877                                             # image row through M1 in image A
    u = (np.arange(2000) + 0.5) * 0.1 - 100
    v = (j + 0.5) * 0.1 - 100
    a = img[j]
    P = apply_transform(frames.F_CK_from_Det("A"), np.stack([u, np.full_like(u, v), 0 * u], 1))
    d = P - SA
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    a_sphere = config.SPHERE_DENSITY_g_cm3 * line_sphere_chord(SA, d, np.zeros(3), 50) / 10
    ax.plot(u, a_sphere, color="gray", lw=2.5, label="sphere only (hand formula)")
    ax.plot(u, a, color=CA, lw=1.2, label="DRR row, with the three balls")
    ax.set_xlim(-60, 60); ax.set_xlabel("$u$ (mm)"); ax.set_ylabel(r"DRR value $\sum_i d_i x_i$ (g/cm$^2$)")
    ax.set_title(f"(a) one image row ($v={v:.1f}$ mm) through M1 in image A")
    ax.legend(frameon=False, loc="upper left")
    ax.annotate("ball adds\n$\\approx 7.6\\,l_{\\rm ball}/10\\approx 1.5$", xy=(21.3, a.max()), xytext=(-55, a.max() * 0.62),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    # (b) signed path length through a closed surface
    bx.add_patch(plt.Polygon([[1.5, 0.5], [3.7, 0.2], [4.6, 1.4], [3.2, 2.3], [1.6, 1.8]], fc="#dddddd", ec="k"))
    bx.add_patch(plt.Polygon([[2.3, 0.9], [2.9, 0.8], [3.0, 1.4], [2.4, 1.5]], fc="white", ec="k", ls="--"))
    bx.annotate("", xy=(5.2, 1.15), xytext=(0.2, 1.15), arrowprops=dict(arrowstyle="->", color=CA, lw=1.5))
    bx.text(0.2, 1.3, "ray from $S$", color=CA)
    for x, c in [(1.58, BAD), (2.33, OK), (2.98, BAD), (4.52, OK)]:
        bx.plot(x, 1.15, "o", color=c, ms=7)
    bx.text(1.0, 1.0, "$-t_1$ entry", color=BAD, ha="center", va="top", fontsize=9)
    bx.text(2.6, 0.5, "$+t_2$ exit (cavity)", color=OK, ha="center", va="top", fontsize=9)
    bx.text(3.05, 1.65, "$-t_3$\nentry (cavity)", color=BAD, ha="center", fontsize=9)
    bx.text(4.65, 1.6, "$+t_4$\nexit", color=OK, ha="center", fontsize=9)
    bx.text(2.7, -0.55, r"length in material $=t_4-t_3+t_2-t_1$", ha="center")
    bx.text(2.7, -0.95, "exit face adds $t$, entry face subtracts $t$:\nno sorting of the hits is needed", ha="center", fontsize=9)
    bx.set_xlim(0, 5.4); bx.set_ylim(-1.3, 2.6); bx.axis("off")
    bx.set_title("(b) one ray against a closed triangle surface")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task5_profile.png"), dpi=150); plt.close(fig)


# ---------------------------------------------------------------- Task 6: method figure + errors
def fig_task6_method():
    img = load("drr_vertebra_A.npy")
    a = img
    res = localize.localize_markers(img)
    # the marker lying on the bone edge/slope is the instructive one: pick the one with the weakest score
    n = int(np.argmin(res["score"]))
    k0, j0 = int(res["peak_px"][n, 0]), int(res["peak_px"][n, 1])
    h = 120
    sl = (slice(j0 - h, j0 + h + 1), slice(k0 - h, k0 + h + 1))
    ext = [k0 - h, k0 + h + 1, j0 - h, j0 + h + 1]
    score = localize.blob_score(a)
    excess, K, J = localize._excess_window(a, k0, j0)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
    ax = axs[0]
    ax.imshow(a[sl], origin="lower", cmap="gray_r", extent=ext)
    ax.set_title("(a) DRR value $a=\\sum_i d_i x_i$\n(vertebra, detector A)")
    ax = axs[1]
    ax.imshow(a[sl], origin="lower", cmap="gray_r", extent=ext, alpha=.35)
    d, b = localize.OFFSET_PX, localize.BOX_PX
    for dk, dj, c in ((0, 0, CA), (d, 0, "#1565c0"), (-d, 0, "#1565c0"), (0, d, "#1565c0"), (0, -d, "#1565c0")):
        ax.add_patch(Rectangle((k0 + dk - b / 2, j0 + dj - b / 2), b, b, fill=False, ec=c, lw=2))
    ax.text(k0, j0 + 2 * d - 8, "centre box", color=CA, ha="center")
    ax.text(k0 + 2 * d - 5, j0 - d - 8, "4 comparison\nboxes", color="#1565c0", ha="center", va="top", fontsize=8)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title("(b) score = smallest of the four\n(centre box $-$ box 3.5 mm away)")
    ax = axs[2]
    im = ax.imshow(score[sl], origin="lower", cmap="viridis", extent=ext)
    ax.set_title("(c) score map: a ball is a peak,\na bone edge is not")
    plt.colorbar(im, ax=ax, fraction=.046)
    ax = axs[3]
    kk, jj = K[0, 0], J[0, 0]
    wext = [kk, K[0, -1] + 1, jj, J[-1, 0] + 1]
    e = np.where(excess > 0.5 * excess.max(), excess, np.nan)
    ax.imshow(excess, origin="lower", cmap="gray_r", extent=wext)
    ax.imshow(e, origin="lower", cmap="autumn", extent=wext, alpha=.85)
    ax.plot(*res["centroid_px"][n], "k+", ms=14, mew=2, label="weighted centroid")
    ax.plot(*res["area_px"][n], "bx", ms=9, mew=2, label="area centroid")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("(d) background plane removed;\npixels above half the peak (orange)")
    for ax in axs:
        ax.set_xlabel("$i_1$ (px)"); ax.set_ylabel("$i_2$ (px)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task6_method.png"), dpi=150); plt.close(fig)


def localization_errors():
    stl, t4 = load("markers_stl_projected.npz"), load("markers_projected_task4.npz")
    out = {}
    for phase, truth in (("sphere", t4), ("vertebra", stl)):
        for pose in "AB":
            res = localize.localize_markers(load(f"drr_{phase}_{pose}.npy"))
            tr = truth[f"Img{pose}_px"]
            for m in ("peak_px", "centroid_px", "area_px"):
                dd = np.linalg.norm(res[m][:, None] - tr[None], axis=2)
                out[(phase, pose, m)] = dd.min(axis=1) / 10.0     # mm
    return out


def fig_task6_errors():
    err = localization_errors()
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    names = {"peak_px": "step 1: peak (pixel)", "centroid_px": "step 2: weighted centroid", "area_px": "area centroid"}
    cols = {"peak_px": "#9e9e9e", "centroid_px": CA, "area_px": CB}
    groups = [(p, q) for p in ("sphere", "vertebra") for q in "AB"]
    w = 0.26
    for i, m in enumerate(names):
        for gi, (p, q) in enumerate(groups):
            ax.scatter(np.full(3, gi + (i - 1) * w), err[(p, q, m)], color=cols[m], s=30, zorder=3,
                       label=names[m] if gi == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{p}\ndetector {q}" for p, q in groups])
    ax.set_ylabel("localization error (mm, log)")
    ax.axhline(0.1, color="k", lw=.6, ls=":")
    ax.text(3.45, 0.105, "1 pixel", ha="right", va="bottom", fontsize=8)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=.25)
    ax.set_title("Distance of each found center from the true projection (3 markers per group)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task6_errors.png"), dpi=150); plt.close(fig)
    return err


# ---------------------------------------------------------------- Task 8
def fig_task8():
    proj = load("markers_projected_task4.npz")
    A, B = proj["DetA_uv_mm"], proj["DetB_uv_mm"]
    fig = plt.figure(figsize=(15, 4.8))
    ax = fig.add_subplot(1, 3, 1)
    rem = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            M, rem[i, j], _ = reconstruct_point(A[i], B[j])
            true = i == j
            ax.plot(M[0], M[1], "o", mfc=OK if true else "none", mec=OK if true else BAD, ms=9 if true else 7, mew=1.5, zorder=4)
    for i in range(3):
        for pose, S, uv, c in (("A", SA, A, CA), ("B", SB, B, CB)):
            T, _ = back_projection_line(uv[i], pose)
            end = T - 0.2 * (S - T)
            ax.plot(*np.array([S, end])[:, :2].T, color=c, lw=1)
    ax.plot(*SA[:2], "o", color=CA)
    ax.plot(*SB[:2], "o", color=CB)
    ax.text(SA[0], SA[1] + 40, "$S_A$", color=CA, ha="center")
    ax.text(SB[0], SB[1] + 40, "$S_B$", color=CB, ha="center")
    ax.set_xlim(-150, 150); ax.set_ylim(-150, 150)
    ax.set_aspect("equal")
    ax.set_title("(a) top view of the 6 back-projection lines:\nthe 9 crossings (green true, red false pairing)")
    ax.set_xlabel("$x_{CK}$ (mm)"); ax.set_ylabel("$y_{CK}$ (mm)")
    ax.invert_xaxis()
    bx = fig.add_subplot(1, 3, 2)
    im = bx.imshow(rem, cmap="Reds", vmin=0, vmax=20)
    for i in range(3):
        for j in range(3):
            bx.text(j, i, f"{rem[i, j]:.1f}", ha="center", va="center", color="w" if rem[i, j] > 11 else "k",
                    fontweight="bold" if i == j else "normal")
    bx.set_xticks(range(3)); bx.set_yticks(range(3))
    bx.set_xticklabels(["B$_1$", "B$_2$", "B$_3$"]); bx.set_yticklabels(["A$_1$", "A$_2$", "A$_3$"])
    bx.set_title("(b) REM (mm) of every pair of points\n(A point $i$, B point $j$)")
    for s in bx.spines.values():
        s.set_visible(False)
    cx = fig.add_subplot(1, 3, 3)
    r = correspond(A, B)
    c1, c2 = r["rem"]["cost"], r["fit"]["cost"]
    x = np.arange(6); w = .38
    cx.bar(x - w / 2, np.maximum(c1, 1e-3), w, color=CA, label="method 1: sum of REM")
    cx.bar(x + w / 2, np.maximum(c2, 1e-3), w, color=CB, label="method 2: rigid-fit error")
    cx.set_yscale("log"); cx.set_ylim(1e-3, 300)
    cx.text(0, 2e-3, r"$\approx 10^{-13}$", ha="center", fontsize=8)
    cx.set_xticks(x); cx.set_xticklabels(["".join(str(q + 1) for q in p) for p in PERMUTATIONS], fontsize=9)
    cx.set_xlabel("pairing: B points matched with A points 1, 2, 3")
    cx.set_ylabel("cost (mm, log, floor $10^{-3}$)")
    cx.legend(frameon=False, fontsize=8, loc="upper left")
    cx.grid(axis="y", alpha=.25)
    cx.set_title("(c) cost of all $3!=6$ pairings\n(true projections, B in true order)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task8_pairings.png"), dpi=150); plt.close(fig)
    return rem, c1, c2


def fig_task8_matched():
    """Task 6 -> 8 -> 7 on the vertebra DRRs: unordered points found by Task 6, colored by matched pair."""
    loc = load("localized") if False else load("markers_localized_task6.npz")
    uvA, uvB = loc["vertebra_A_uv_mm"], loc["vertebra_B_uv_mm"]
    r = correspond(uvA, uvB)
    cols = ["#1b9e77", "#d95f02", "#7570b3"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, uv, name in ((axs[0], uvA, "A"), (axs[1], uvB, "B")):
        ax.imshow(load(f"drr_vertebra_{name}.npy"), origin="lower", cmap="gray_r",
                  extent=[-100, 100, -100, 100], alpha=.55)
        for i in range(3):
            q = uv[i] if name == "A" else uv[r["perm"][i]]
            ax.plot(*q, "o", mfc="none", mec=cols[i], ms=16, mew=2.5)
            ax.text(q[0] + 8, q[1] + 6, f"pair {i + 1}", color=cols[i], fontweight="bold")
        ax.set_title(f"detector {name}: points found in Task 6,\ncolored by the pairing chosen in Task 8")
        ax.set_xlabel("$u$ (mm)"); ax.set_ylabel("$v$ (mm)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "task8_matched.png"), dpi=150); plt.close(fig)
    return r


if __name__ == "__main__":
    fig_task5_geometry(); fig_task5_profile(); fig_task6_method()
    err = fig_task6_errors()
    for k, v in err.items():
        print(k, np.round(v, 4))
    rem, c1, c2 = fig_task8()
    print("REM matrix\n", np.round(rem, 2)); print("cost1", np.round(c1, 3)); print("cost2", np.round(c2, 3))
    r = fig_task8_matched()
    print(r["perm"], r["fit"]["labels"], np.round(r["REM"], 4))
    print("figures written to", OUT)
