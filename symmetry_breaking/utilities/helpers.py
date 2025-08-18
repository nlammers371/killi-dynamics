import numpy as np
from scipy.signal import find_peaks

def count_nodal_peaks_periodic(
    N_profile,
    min_sep=100,        # absolute height threshold = median + k_height * noise
    k_prom=None,         # prominence threshold = k_prom * noise
    height_thresh=0,
):
    """
    Count peaks on a 1D *periodic* profile without smoothing.
    Returns (n_peaks, peak_indices) with indices in [0, N).
    """
    N = np.asarray(N_profile, float)
    L = len(N)


    # --- Handle periodic boundary by tiling 3× and keeping the middle window ---
    N_ext = np.concatenate([N, N, N])             # length 3L
    peaks_ext, props = find_peaks(
        N_ext,
        height=height_thresh,
        prominence=k_prom,
        distance=min_sep,
        width=1
    )

    # Keep only peaks whose centers fall in the middle copy [L, 2L)
    in_mid = (peaks_ext >= L) & (peaks_ext < 2*L)
    peaks_mid = peaks_ext[in_mid] - L             # map back to [0, L)

    # --- Deduplicate modulo-L just in case (rare) ---
    # Sort and drop peaks closer than min_sep on the circle
    if len(peaks_mid) == 0:
        return 0, np.array([], dtype=int)

    # Circular greedy pruning
    peaks_sorted = np.sort(peaks_mid)
    keep = []
    for p in peaks_sorted:
        if all(min((p - k) % L, (k - p) % L) >= min_sep for k in keep):
            keep.append(p)

    # Final sanity: enforce min_sep on circle (merge by keeping higher peak)
    # (Usually unnecessary because distance is enforced on the extended array.)

    return len(keep), np.array(keep, dtype=int)