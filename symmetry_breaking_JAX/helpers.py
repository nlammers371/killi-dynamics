from symmetry_breaking_JAX.models.JAX_NL_1D import run_1d, Params1D, BlipSet1D, build_initial_state_1d
import jax.numpy as jnp
import numpy as np
import os
from pathlib import Path
from typing import Union

def split_numeric_and_static(params_list):
    numeric = {}
    static = {}
    for k in params_list[0].keys():
        values = [p[k] for p in params_list]
        # Check if numeric (int/float/np.number)
        if np.issubdtype(type(values[0]), np.number):
            numeric[k] = jnp.array(values)
        else:
            static[k] = values
    return numeric, static

# --------------------------------------------------
# Wrap for JAX
# --------------------------------------------------
def single_run(dim_p, x, dx, L, T, nx, n_save_points=121):

    # Build Params1D dataclass from dim_p dict
    p = Params1D(
        D_N=dim_p["D_N"], D_L=dim_p["D_L"],
        sigma_N=dim_p["sigma_N"], sigma_L=dim_p["sigma_L"],
        mu_N=dim_p["mu_N"], mu_L=dim_p["mu_L"],
        n=2, p=2, alpha=1.0,
        K_A=dim_p["K_A"], K_NL=dim_p["K_NL"], K_I=dim_p["K_I"],
        Lx=L, dx=dx, bc="neumann",
        tau_impulse=60.0,
        sigma_t_direct=0.1,
    )

    # No stochastic blips for now
    blips = BlipSet1D.empty()

    # Initial condition
    y0 = build_initial_state_1d(
        x,
        N_mode="gaussian", N_amp=dim_p["N_amp"], N_sigma=dim_p["N_sigma"],
        L_mode="constant", L_amp=dim_p["L_value"],
        rho_value=0.1,
    )

    # Run simulation
    save_ts = jnp.linspace(0, T, n_save_points)  # every ~5 min
    _, _, ys = run_1d(p, blips, T=T, nx=nx, save_ts=save_ts, y0_override=y0)

    # Split into nodal and lefty
    N = ys[:, :nx]  # shape (n_times, nx)
    L = ys[:, nx:]  # shape (n_times, nx)

    return N, L


def make_param_dicts(param_grid, grid_type="grid", n_samples=1000, seed=42):
    """
    grid_type: "grid" | "random" | "lhs"
      - "grid": full Cartesian product (be careful with size)
      - "random": uniform random over the Cartesian list of combos (your current behavior)
      - "lhs": Latin Hypercube over the *indices* of each parameter list (minimal-change & efficient)
    """
    rng = np.random.default_rng(seed)
    keys = list(param_grid.keys())
    values = [np.asarray(param_grid[k]) for k in keys]

    if grid_type == "grid":
        import itertools
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    if grid_type == "random":
        # sample combos uniformly at random from full Cartesian product
        sizes = [len(v) for v in values]
        total = np.prod(sizes, dtype=int)
        if n_samples > total:
            n_samples = total
        picks = rng.choice(total, size=n_samples, replace=False)
        # decode linear index to mixed radix
        out = []
        for p in picks:
            idxs = []
            rem = p
            for s in reversed(sizes):
                idxs.append(rem % s)
                rem //= s
            idxs = list(reversed(idxs))
            out.append({k: values[i][idxs[i]].item() for i, k in enumerate(keys)})
        return out

    if grid_type == "lhs":
        n = n_samples
        idx_matrix = []
        for v in values:
            idx = _lhs_indices(len(v), int(n), rng)   # stratified indices for this dim
            idx_matrix.append(idx)
        idx_matrix = np.stack(idx_matrix, axis=1)  # shape (n_samples, n_params)
        # Build param dicts by taking the i-th pick from each dimension
        out = []
        for row in idx_matrix:
            out.append({k: values[j][row[j]].item() for j, k in enumerate(keys)})
        return out

    raise ValueError(f"Unknown grid_type: {grid_type}")

def ensure_output_dir(path: Union[str, os.PathLike]):
    os.makedirs(path, exist_ok=True)
    return path

def _lhs_indices(n_levels, n_samples, rng):
    """
    Latin Hypercube in [0,1): one stratified sample per bin, permuted.
    Then map to integer indices in [0, n_levels-1].
    """
    # One point per stratum, jittered
    u = (np.arange(n_samples) + rng.random(n_samples)) / n_samples  # shape (n_samples,)
    rng.shuffle(u)  # permute order for this dimension
    # Map to discrete indices
    idx = np.floor(u * n_levels).astype(int)
    # guard for edge case u==1.0 due to float
    idx = np.clip(idx, 0, n_levels - 1)
    return idx



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