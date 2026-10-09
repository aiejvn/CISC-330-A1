"""
localize.py - Task 6: Marker Localization.

Find the centers of the three marker shadows in a DRR (pixels) and convert them
to detector coordinates (mm). The DRR array is img[j, k] = sum d_i x_i (Task 5) with k = image
column = i1 and j = image row = i2; pixel (k, j) has its middle at (k + 0.5, j + 0.5)
in the image frame, i.e. u = (k + 0.5) * 0.1 - 100 mm (see drr.py).

Work on the DRR value a = sum d_i x_i itself: a steel ball adds a compact bump of
about 1.5 to a (2 mm of steel, 7.85 g/cm^3) to whatever lies behind it.

Step 1, detection (method "peak"). A marker is a spot that is higher than the
ground on ALL FOUR sides. For every pixel we take the mean of a small box
around it (21 x 21 pixels = 2.1 mm, smaller than a shadow) and subtract the same
box mean taken d = 35 pixels to the left, right, up and down. The score of the
pixel is the SMALLEST of the four differences. A straight bone edge or a smooth
slope is high on one side only, so its score is about 0; a ball is high on every
side, so its score is large. The n best-scoring pixels (at least 8 mm apart)
are the markers. This is accurate to about a pixel.

Step 2, refinement (method "centroid"). In a 61 x 61 pixel window around each
peak, fit a plane to the border pixels (the local background), subtract it, keep
the pixels whose excess is above half of the peak excess and take the excess-
weighted center of gravity. That gives the center in fractions of a pixel.

Markers are identical, so the order of the returned points is arbitrary
(best score first). Putting them in order is Task 8 (correspondences).

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import numpy as np
import config, frames
from geometry import apply_transform

BOX_PX = 21          # side of the small box (2.1 mm) that averages the center
OFFSET_PX = 35       # distance (3.5 mm) of the four comparison boxes
MIN_SEPARATION_PX = 80   # two markers are never closer than 8 mm on a detector (suppression radius)
WINDOW_HALF_PX = 30  # refinement window is 61 x 61 pixels (6.1 mm). A shadow is about 34 px wide (half-max), so the
                     # window needs the shadow plus a free 7 px frame: at 51 px the frame touches the shadow (sphere
                     # error doubles); above ~71 px the plane no longer fits the curved bone (vertebra error 0.7 px at
                     # 61 px, 1.2 px at 101 px, 1.7 px at 151 px). Measured against the true centers, Task 5/6.
BORDER_PX = 7        # width of the frame of the window used for the background plane


def box_mean(a, win):
    """Mean over a win x win box around every pixel (odd win), by an integral image; edges mirrored."""
    h = win // 2
    c = np.cumsum(np.cumsum(np.pad(a, h, mode="reflect"), axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    n_j, n_k = a.shape
    total = (c[win:win + n_j, win:win + n_k] - c[:n_j, win:win + n_k]
             - c[win:win + n_j, :n_k] + c[:n_j, :n_k])
    return total / win ** 2


def _shift(m, dj, dk):
    """m shifted so that result[j, k] = m[j + dj, k + dk] (edge values repeated)."""
    q = max(abs(dj), abs(dk))
    p = np.pad(m, q, mode="edge")
    n_j, n_k = m.shape
    return p[q + dj:q + dj + n_j, q + dk:q + dk + n_k]


def blob_score(a):
    """Smallest difference between the center box and the four boxes OFFSET_PX away (left/right/up/down)."""
    m = box_mean(a, BOX_PX)
    d = OFFSET_PX
    return np.min([m - _shift(m, dj, dk) for dj, dk in ((0, d), (0, -d), (d, 0), (-d, 0))], axis=0)


def find_peaks(score, n_markers=3):
    """The n best local maxima of the score, at least MIN_SEPARATION_PX apart. Returns [(k, j, score), ...]."""
    s = score.copy()
    peaks = []
    for _ in range(n_markers):
        j, k = np.unravel_index(np.argmax(s), s.shape)
        peaks.append((int(k), int(j), float(s[j, k])))
        s[max(0, j - MIN_SEPARATION_PX):j + MIN_SEPARATION_PX + 1,
          max(0, k - MIN_SEPARATION_PX):k + MIN_SEPARATION_PX + 1] = -np.inf
    return peaks


def _excess_window(a, k0, j0):
    """Absorbance in the window around pixel (k0, j0) minus a plane fitted to the window border; with pixel indices K, J."""
    n_j, n_k = a.shape
    j_lo, j_hi = max(0, j0 - WINDOW_HALF_PX), min(n_j, j0 + WINDOW_HALF_PX + 1)
    k_lo, k_hi = max(0, k0 - WINDOW_HALF_PX), min(n_k, k0 + WINDOW_HALF_PX + 1)
    win = a[j_lo:j_hi, k_lo:k_hi]
    J, K = np.mgrid[j_lo:j_hi, k_lo:k_hi]
    border = ((J < j_lo + BORDER_PX) | (J >= j_hi - BORDER_PX)
              | (K < k_lo + BORDER_PX) | (K >= k_hi - BORDER_PX))
    X = np.stack([np.ones(border.sum()), K[border], J[border]], axis=1)
    coef, *_ = np.linalg.lstsq(X, win[border], rcond=None)      # background plane p0 + p1 k + p2 j
    return win - (coef[0] + coef[1] * K + coef[2] * J), K, J


def refine_centroid(a, k0, j0):
    """
    Center of gravity of the shadow around pixel (k0, j0), in image-frame pixels (i1, i2),
    after removing a plane fitted to the border of the window. Every pixel above half of
    the peak excess counts with a weight equal to its excess.
    """
    excess, K, J = _excess_window(a, k0, j0)
    w = np.where(excess > 0.5 * excess.max(), excess, 0.0)
    return (np.sum(w * (K + 0.5)) / w.sum(), np.sum(w * (J + 0.5)) / w.sum())   # (i1, i2)


def refine_area_centroid(a, k0, j0):
    """
    Centroid of the AREA of the shadow: same background removal, then the region of pixels above
    half of the peak excess that is connected to the peak pixel, every pixel counting equally
    (no intensity weights), so only the shape of the shadow matters. (i1, i2) in pixels.
    """
    excess, K, J = _excess_window(a, k0, j0)
    mask = excess > 0.5 * excess.max()
    seed = (j0 - J[0, 0], k0 - K[0, 0])                       # peak pixel inside the window
    region = np.zeros_like(mask)
    stack = [seed] if mask[seed] else [np.unravel_index(np.argmax(excess), excess.shape)]
    while stack:                                              # flood fill, 4-connected
        r, c = stack.pop()
        if region[r, c] or not mask[r, c]:
            continue
        region[r, c] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]:
                stack.append((rr, cc))
    return (np.mean(K[region] + 0.5), np.mean(J[region] + 0.5))   # (i1, i2)


def localize_markers(img, n_markers=3):
    """
    Find the marker shadows in one DRR.
    Returns a dict with arrays of shape (n, 2) in image-frame pixels (i1, i2):
      "peak_px"     pixel-accurate result of step 1,
      "centroid_px" sub-pixel result of step 2 (intensity-weighted),
      "area_px"     sub-pixel centroid of the shadow's area (unweighted, connected region),
    and "score" (n,): detection scores, best first, plus "next_score": the score of
    the (n+1)-th best candidate (it must be far below the markers' scores).
    """
    a = np.asarray(img, float)
    score = blob_score(a)
    peaks = find_peaks(score, n_markers + 1)
    found, extra = peaks[:n_markers], peaks[n_markers]
    return {"peak_px": np.array([[k + 0.5, j + 0.5] for k, j, _ in found]),
            "centroid_px": np.array([refine_centroid(a, k, j) for k, j, _ in found]),
            "area_px": np.array([refine_area_centroid(a, k, j) for k, j, _ in found]),
            "score": np.array([s for *_, s in found]),
            "next_score": extra[2]}


def image_px_to_detector_mm(px):
    """(i1, i2) image pixels -> (u, v) detector mm (inverse of projector.detector_to_image)."""
    px = np.asarray(px, float)
    ipw = np.concatenate([px, np.zeros(px.shape[:-1] + (1,))], axis=-1)
    return apply_transform(frames.F_Det_from_Img(), ipw)[..., :2]
