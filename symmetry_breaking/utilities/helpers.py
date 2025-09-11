import numpy as np
from scipy.signal import find_peaks

def nd_to_dim(nd_params, anchors):
    """
    Convert a dictionary of ND params into dimension-full params.

    Parameters
    ----------
    nd_params : dict
        ND parameters (beta_a, beta_r, rho_mu, delta,
                       kappa_I, kappa_NL, a_amp, r_value, etc.)
    anchors : dict
        Absolute anchors:
            mu_N : float  [1/s]
            D_N  : float  [µm^2/s]
            K_A  : float  [concentration units]
            (optionally: K_I, K_NL, N_sigma, etc. if you want to override)

    Returns
    -------
    dict : dimension-full parameters for NodalLeftyField1D
    """
    mu_N = anchors["mu_N"]
    D_N = anchors["D_N"]
    K_A = anchors["K_A"]

    out = {}

    # Kinetics
    out["sigma_N"] = nd_params.get("beta_a", 1.0) * mu_N * K_A
    out["sigma_L"] = nd_params.get("beta_r", 1.0) * mu_N * K_A
    out["mu_N"] = mu_N
    out["mu_L"] = nd_params.get("rho_mu", 1.0) * mu_N

    # Diffusion
    out["D_N"] = D_N
    out["D_L"] = nd_params.get("delta", 1.0) * D_N

    # Binding constants
    out["K_A"] = K_A
    out["K_I"] = nd_params.get("kappa_I", 1.0) * K_A
    out["K_NL"] = nd_params.get("kappa_NL", 1.0) * K_A

    # Hill exponents
    for key in ["n", "m", "p", "q"]:
        if key in nd_params:
            out[key] = nd_params[key]

    # Initial conditions
    if "a_amp" in nd_params:
        out["N_amp"] = nd_params["a_amp"] * K_A
    if "r_value" in nd_params:
        out["L_value"] = nd_params["r_value"] * K_A

    # Pass through any extras (init modes, etc.)
    for k in ["L_init", "N_sigma"]:
        if k in anchors:
            out[k] = anchors[k]

    return out


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