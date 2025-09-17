import numpy as np
from scipy.signal import find_peaks

def nd_to_dim(nd_params: dict, anchors: dict) -> dict:
    """
    Robust ND → dimensional conversion.

    Required anchors: mu_N [1/s], D_N [µm^2/s], K_A [conc]
    Recommend anchors: sigma_N_ref [conc/s] (fixes β_a), optional K_I (absolute)
    """
    mu_N = anchors["mu_N"]
    D_N  = anchors["D_N"]
    K_A  = anchors["K_A"]

    # --- β_a and σ_N ---
    if "sigma_N_ref" in anchors:
        sigma_N = float(anchors["sigma_N_ref"])
        beta_a  = sigma_N / (mu_N * K_A)
    else:
        beta_a  = float(nd_params.get("beta_a", 1.0))
        sigma_N = beta_a * mu_N * K_A

    # --- β_r and σ_L ---
    beta_r = float(nd_params.get("beta_r", beta_a))  # default: match σ_L=σ_N
    sigma_L = beta_r * mu_N * K_A

    # --- decay & diffusion ---
    rho_mu = float(nd_params.get("rho_mu", anchors.get("rho_mu", 1.0)))
    mu_L = rho_mu * mu_N
    delta = float(nd_params.get("delta", 1.0))
    D_L = delta * D_N

    # --- binding constants ---
    kappa_NL = float(nd_params.get("kappa_NL", 1.0))
    K_NL = kappa_NL * K_A
    if "K_I" in anchors:         # absolute override (strong-binding)
        K_I = float(anchors["K_I"])
    else:
        kappa_I = float(nd_params.get("kappa_I", 1e-6))
        K_I = kappa_I * K_A

    # --- initials ---
    out = {
        "sigma_N": sigma_N,
        "sigma_L": sigma_L,
        "mu_N": mu_N,
        "mu_L": mu_L,
        "D_N": D_N,
        "D_L": D_L,
        "K_A": K_A,
        "K_I": K_I,
        "K_NL": K_NL,
    }
    if "a_amp" in nd_params:
        out["N_amp"] = float(nd_params["a_amp"]) * K_A
    if "r_value" in nd_params:
        out["L_value"] = float(nd_params["r_value"]) * K_A

    # pass-through
    if "L_init" in anchors: out["L_init"] = anchors["L_init"]
    if "N_sigma" in anchors: out["N_sigma"] = anchors["N_sigma"]
    for k in ("n","m","p","q"):
        if k in nd_params: out[k] = int(nd_params[k])

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