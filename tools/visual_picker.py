"""
visual_picker.py - Task 6 visual localization: open a saved DRR, move the cursor over a marker (the status bar
shows image pixel (i1, i2), detector (u, v) in mm and the DRR value), click to record the marker center.

Usage (from the A1 folder):
    python tools/visual_picker.py sphere A          # phase: sphere | vertebra, pose: A | B

Mouse / keys
    left click   record a marker center (up to 3)       right click or u   undo the last click
    scroll       zoom around the cursor                  r                  reset the view
    s            save picks to outputs/visual/picks_<phase>_<pose>.csv (also saved on close)
    t            show / hide the TRUE projected centers (hidden by default, so the reading is blind)
    a            show / hide the ALGORITHM's centers (Task 6 result)
Use the zoom tool or the scroll wheel: the dark disc is about 60 px wide, and 1 px = 0.1 mm.
tools/compare_visual_readings.py uses the saved picks (when they exist) for the comparison table.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

TRUTH = {"sphere": "markers_projected_task4.npz", "vertebra": "markers_stl_projected.npz"}


def picks_path(phase, pose):
    return os.path.join(config.OUTPUT_DIR, "visual", f"picks_{phase}_{pose}.csv")


def main(phase, pose):
    img = np.load(os.path.join(config.OUTPUT_DIR, f"drr_{phase}_{pose}.npy"))            # [j, k]
    true_px = np.load(os.path.join(config.OUTPUT_DIR, TRUTH[phase]))[f"Img{pose}_px"]
    algo_px = np.load(os.path.join(config.OUTPUT_DIR, "markers_localized_task6.npz"))[f"{phase}_{pose}_px"]
    n = img.shape[0]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap="gray_r", origin="lower", interpolation="nearest", extent=[0, n, 0, n])
    ax.set_xlabel("i1 (px)"); ax.set_ylabel("i2 (px)")
    ax.set_title(f"{phase} DRR, pose {pose}: click the center of each marker (3)")
    picks, artists = [], []
    overlays = {"t": ax.plot(true_px[:, 0], true_px[:, 1], "o", mfc="none", mec="limegreen", ms=18, visible=False)[0],
                "a": ax.plot(algo_px[:, 0], algo_px[:, 1], "+", color="red", ms=16, visible=False)[0]}

    def value_at(x, y):
        return img[int(np.clip(y, 0, n - 1)), int(np.clip(x, 0, n - 1))]

    ax.format_coord = lambda x, y: (f"i1={x:7.1f} i2={y:7.1f} px | u={x * config.PIXEL_SIZE_mm - 100:7.2f} "
                                    f"v={y * config.PIXEL_SIZE_mm - 100:7.2f} mm | sum d x = {value_at(x, y):.3f} g/cm^2")

    def redraw():
        for a in artists:
            a.remove()
        artists.clear()
        for i, (x, y) in enumerate(picks):
            artists.append(ax.plot(x, y, "x", color="tab:blue", ms=12, mew=2)[0])
            artists.append(ax.annotate(str(i + 1), (x, y), xytext=(8, 8), textcoords="offset points", color="tab:blue"))
        ax.set_title(f"{phase} DRR, pose {pose}: {len(picks)} of 3 markers recorded "
                     f"({'click the next center' if len(picks) < 3 else 'press s to save'})")
        fig.canvas.draw_idle()

    def save():
        os.makedirs(os.path.dirname(picks_path(phase, pose)), exist_ok=True)
        with open(picks_path(phase, pose), "w") as f:
            f.write("i1,i2\n" + "".join(f"{x:.1f},{y:.1f}\n" for x, y in picks))
        print("saved", picks_path(phase, pose), picks)

    def on_click(e):
        if e.inaxes is not ax or getattr(fig.canvas.toolbar, "mode", ""):      # ignore clicks while the zoom / pan tool is active
            return
        if e.button == 1 and len(picks) < 3:
            picks.append((round(e.xdata, 1), round(e.ydata, 1)))
        elif e.button == 3 and picks:
            picks.pop()
        redraw()

    def on_key(e):
        if e.key in overlays:
            overlays[e.key].set_visible(not overlays[e.key].get_visible()); fig.canvas.draw_idle()
        elif e.key == "u" and picks:
            picks.pop(); redraw()
        elif e.key == "s":
            save()
        elif e.key == "r":
            ax.set_xlim(0, n); ax.set_ylim(0, n); fig.canvas.draw_idle()

    def on_scroll(e):
        if e.inaxes is not ax:
            return
        s = 0.7 if e.button == "up" else 1 / 0.7
        x0, x1, y0, y1 = *ax.get_xlim(), *ax.get_ylim()
        ax.set_xlim(e.xdata - (e.xdata - x0) * s, e.xdata + (x1 - e.xdata) * s)
        ax.set_ylim(e.ydata - (e.ydata - y0) * s, e.ydata + (y1 - e.ydata) * s)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("close_event", lambda e: picks and save())
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in TRUTH or sys.argv[2] not in ("A", "B"):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
