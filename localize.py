"""
localize.py - Task 6: Marker Localization.

Find the centers of the three marker shadows in a DRR (pixels) and convert them
to detector coordinates (mm). The DRR array is img[j, k] = sum d_i x_i (Task 5) with k = image
column = i1 and j = image row = i2; pixel (k, j) has its middle at (k + 0.5, j + 0.5)
in the image frame, i.e. u = (k + 0.5) * 0.1 - 100 mm (see drr.py). The 100 is half the
200 mm detector width (config.DETECTOR_SIZE_mm / 2): the detector is centred on 0, so
its left edge is at -100 mm.

Work on the DRR value a = sum d_i x_i itself: a steel ball adds a compact bump of
about 2.3 to a (3 mm of steel, 7.85 g/cm^3) to whatever lies behind it.

Step 1, detection (method "peak"). A marker is a spot that is higher than the
ground on ALL FOUR sides. For every pixel we take the mean of a small box
around it (21 x 21 pixels = 2.1 mm, smaller than a shadow) and subtract the same
box mean taken d = 45 pixels to the left, right, up and down. The score of the
pixel is the SMALLEST of the four differences. A straight bone edge or a smooth
slope is high on one side only, so its score is about 0; a ball is high on every
side, so its score is large. The n best-scoring pixels (at least 8 mm apart)
are the markers. This is accurate to about a pixel.

Step 2, refinement (method "centroid"). In an 81 x 81 pixel window around each
peak, fit a plane to the border pixels (the local background), subtract it, keep
the pixels whose excess is above half of the peak excess and take the excess-
weighted center of gravity. That gives the center in fractions of a pixel.
Why each part:
  - Background plane: the bone behind the ball slopes; subtracting a plane fitted
    to the frame leaves only the ball's bump, so the slope cannot drag the center.
  - Half of the peak: keeps the shadow itself and drops noise and leftover
    background (the usual half-maximum convention; not tuned).
  - Excess weights: the deepest part of the shadow counts most (center of gravity).
  - 81 px window, 7 px border: tuned against the true centers; a smaller window puts
    the frame on the shadow, a larger one is too curved for a plane (see WINDOW_HALF_PX).

Markers are identical, so the order of the returned points is arbitrary
(best score first). Putting them in order is Task 8 (correspondences).

AI use: written with help from Claude (Anthropic), Oct 2026; reviewed and
tested by the group.
"""
import numpy as np
import config, frames
from geometry import apply_transform

BOX_PX = 21          # side of the small box (2.1 mm) that averages the center
OFFSET_PX = 45       # distance (4.5 mm) of the four comparison boxes
MIN_SEPARATION_PX = 80   # two markers are never closer than 8 mm on a detector (suppression radius)
WINDOW_HALF_PX = 40  # refinement window is 81 x 81 pixels (8.1 mm). A shadow of a 3 mm ball is about 60 px wide (2 x 3 mm
                     # magnified by 2) and its half-max width about 52 px, so the window needs the shadow plus a free
                     # 7 px frame: at 71 px the frame touches the shadow; the larger the window, the less a plane fits the
                     # curved bone (vertebra worst error 1.25 px at 81 px, 1.5 px at 91 px, 1.7 px at 101 px, 2.4 px at
                     # 121 px; the sphere phase stays at 0.05 px). Measured against the true centers, Task 5/6.
BORDER_PX = 7        # width of the frame of the window used for the background plane


def box_mean(a, win):
    """Mean over a win x win box around every pixel (odd win); edges mirrored."""
    h = win // 2                                                   # half width of the box
    n_j, n_k = a.shape
    # add a frame h pixels wide around the image, filled by copying the pixels just inside the
    # edge (like a mirror), so that pixels at the edge still have a full box around them
    p = np.pad(a, h, mode="reflect")
    total = np.zeros_like(a)
    # the box around a pixel covers offsets 0 ... win-1 in each direction of the framed image.
    # Adding the whole image shifted by each offset, one after another, makes total[j, k] the
    # sum of the box around pixel (j, k): every pixel gets its win x win neighbours added to it
    for dj in range(win):
        for dk in range(win):
            total += p[dj:dj + n_j, dk:dk + n_k]
    return total / win ** 2                                        # sum -> mean


def _shift(m, dj, dk):
    """m shifted so that result[j, k] = m[j + dj, k + dk] (edge values repeated)."""
    q = max(abs(dj), abs(dk))                                      # frame width = the biggest shift, so we never run off the image
    p = np.pad(m, q, mode="edge")                                  # add a frame around the image, copying the outermost pixels outward
    n_j, n_k = m.shape
    # cut out a piece of the framed image that is as big as the original, but moved by (dj, dk)
    return p[q + dj:q + dj + n_j, q + dk:q + dk + n_k]


def blob_score(a):
    """Smallest difference between the center box and the four boxes OFFSET_PX away (left/right/up/down)."""
    # a ball is higher than the ground on all four sides; a bone edge or a slope is higher on one side
    # only. Taking the smallest of the four differences keeps the ball and removes the edges.
    m = box_mean(a, BOX_PX)                                        # blurred image: every pixel = average of its small box
    d = OFFSET_PX
    # make four images, each "this pixel minus the pixel d away to the right / left / down / up".
    # Then np.min(..., axis=0) goes through the four images and keeps, for every pixel, the smallest
    # of its four values. (Big only if the pixel is higher than its neighbours in ALL four directions.)
    return np.min([m - _shift(m, dj, dk) for dj, dk in ((0, d), (0, -d), (d, 0), (-d, 0))], axis=0)


def find_peaks(score, n_markers=3):
    """The n best local maxima of the score, at least MIN_SEPARATION_PX apart. Returns [(k, j, score), ...]."""
    s = score.copy()                                               # work on a copy, because we erase parts of it below
    peaks = []          # greedy: take the best pixel, then erase its surroundings so a marker is not found twice
    for _ in range(n_markers):
        # np.argmax gives the position of the largest value counted along the whole image row by row;
        # np.unravel_index turns that single number back into (row, column)
        j, k = np.unravel_index(np.argmax(s), s.shape)
        peaks.append((int(k), int(j), float(s[j, k])))             # int() / float(): turn numpy numbers into normal Python numbers
        # fill a square block around the peak with -infinity so it can never be the largest again
        # (max(0, ...) stops the block from starting at a negative index at the image edge)
        s[max(0, j - MIN_SEPARATION_PX):j + MIN_SEPARATION_PX + 1,
          max(0, k - MIN_SEPARATION_PX):k + MIN_SEPARATION_PX + 1] = -np.inf
    return peaks


def _excess_window(a, k0, j0):
    """Absorbance in the window around pixel (k0, j0) minus a plane fitted to the window border; with pixel indices K, J."""
    # background = the plane through the frame of the window (what the bone behind the ball would be
    # without the ball); subtracting it leaves only the bump of the ball itself
    n_j, n_k = a.shape
    j_lo, j_hi = max(0, j0 - WINDOW_HALF_PX), min(n_j, j0 + WINDOW_HALF_PX + 1)   # window rows, clipped to the image
    k_lo, k_hi = max(0, k0 - WINDOW_HALF_PX), min(n_k, k0 + WINDOW_HALF_PX + 1)   # window columns, clipped
    win = a[j_lo:j_hi, k_lo:k_hi]                                  # the window cut out of the image (not a separate copy)
    # J and K are two tables of the same size as the window: J holds the row number of every pixel,
    # K holds its column number
    J, K = np.mgrid[j_lo:j_hi, k_lo:k_hi]
    # True / False table: True for pixels in the outer BORDER_PX frame of the window.
    # "|" means "or", checked pixel by pixel (too close to the top, bottom, left or right edge)
    border = ((J < j_lo + BORDER_PX) | (J >= j_hi - BORDER_PX)
              | (K < k_lo + BORDER_PX) | (K >= k_hi - BORDER_PX))
    # one row [1, k, j] for every frame pixel (K[border] = the column numbers of just the frame pixels).
    # We want a plane p0 + p1*k + p2*j, so each row says "multiply p0 by 1, p1 by k, p2 by j"
    X = np.stack([np.ones(border.sum()), K[border], J[border]], axis=1)
    # least squares: the p0, p1, p2 that make the plane come closest to the frame pixel values
    coef, *_ = np.linalg.lstsq(X, win[border], rcond=None)      # background plane p0 + p1 k + p2 j
    # put the plane's value at every pixel of the window (K and J are the tables of column / row
    # numbers, so this is done for all pixels at once) and subtract it from the window
    return win - (coef[0] + coef[1] * K + coef[2] * J), K, J


def refine_centroid(a, k0, j0):
    """
    Center of the shadow's brightness: after background removal, take the pixels above half the
    peak and average their positions, weighted by how bright they are. (i1, i2) in pixels.
    """
    excess, K, J = _excess_window(a, k0, j0)
    # keep only the shadow itself (above half of the peak) so that noise and background do not pull the
    # center; the excess is the weight, so the deepest part of the shadow counts most
    # np.where(condition, x, y): for every pixel use x where the condition is True, otherwise y.
    # So pixels at or below half of the peak get weight 0, the others keep their excess as weight
    w = np.where(excess > 0.5 * excess.max(), excess, 0.0)
    # weighted average of the pixel positions (big weight = pulls the center toward that pixel);
    # +0.5 because a pixel's middle is half a pixel past its index
    return (np.sum(w * (K + 0.5)) / w.sum(), np.sum(w * (J + 0.5)) / w.sum())   # (i1, i2)


def refine_area_centroid(a, k0, j0):
    """
    Center of the shadow's shape: after the same background removal, take the connected patch
    of pixels above half the peak and average their positions (no weights). (i1, i2) in pixels.
    """
    excess, K, J = _excess_window(a, k0, j0)
    mask = excess > 0.5 * excess.max()                        # True / False table: "is this pixel above half the peak"
    seed = (j0 - J[0, 0], k0 - K[0, 0])                       # peak pixel inside the window
    region = np.zeros_like(mask)     # connected region only: other bright bone above half the peak is not the ball
    # region starts as an all-False table the same size as the mask; pixels are set to True as we reach them.
    # Start at the peak pixel, or at the brightest pixel of the window if the peak pixel is below half
    stack = [seed] if mask[seed] else [np.unravel_index(np.argmax(excess), excess.shape)]
    # flood fill: take a pixel from the to-do list, mark it, and add its neighbours (up, down, left,
    # right) to the list; pixels that are below half or already marked are skipped
    while stack:
        r, c = stack.pop()
        if region[r, c] or not mask[r, c]:
            continue
        region[r, c] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]:
                stack.append((rr, cc))
    # K[region] = the column numbers of only the marked pixels; their plain average is the center of the area
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
    a = np.asarray(img, float)                             # make sure it is a float array (no copy if it already is)
    score = blob_score(a)                                # step 1: detection (pixel accurate)
    peaks = find_peaks(score, n_markers + 1)               # one extra peak: it must score far below the markers
    found, extra = peaks[:n_markers], peaks[n_markers]     # step 2 (refinement) is done per marker below
    return {"peak_px": np.array([[k + 0.5, j + 0.5] for k, j, _ in found]),
            "centroid_px": np.array([refine_centroid(a, k, j) for k, j, _ in found]),
            "area_px": np.array([refine_area_centroid(a, k, j) for k, j, _ in found]),
            "score": np.array([s for *_, s in found]),          # every item of found is (k, j, score); "*_" ignores k and j, s is the score
            "next_score": extra[2]}


def image_px_to_detector_mm(px):
    """(i1, i2) image pixels -> (u, v) detector mm (inverse of projector.detector_to_image)."""
    px = np.asarray(px, float)
    # the transform works on 3D points, so add z = 0 to every (i1, i2): make a table of zeros that is
    # the same as px but with one value per point, and glue it on as a third coordinate
    # (px.shape[:-1] = the shape of px without its last number; axis=-1 = glue on at the end)
    ipw = np.concatenate([px, np.zeros(px.shape[:-1] + (1,))], axis=-1)
    # apply the transform, then [..., :2] keeps only the first two coordinates (u, v) of every point
    return apply_transform(frames.F_Det_from_Img(), ipw)[..., :2]
