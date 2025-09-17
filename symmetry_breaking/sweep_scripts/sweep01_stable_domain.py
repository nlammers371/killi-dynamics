from symmetry_breaking.utilities.helpers import nd_to_dim
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path

from symmetry_breaking.models.NL_field_1D import NodalLeftyField1D
from symmetry_breaking.models.trackers import NodalROITracker
from symmetry_breaking.models.sweep import (
    make_param_dicts,
    ensure_output_dir,
    run_and_save_single_simulation,
)


# --------------------------------------------------
# Sweep script
# --------------------------------------------------
if __name__ == "__main__":
    sweep_name = "sweep01_nd_stable_domain"
    root = Path(
        "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/symmetry_breaking/pde/sweeps/"
    )
    output_dir = ensure_output_dir(root / sweep_name)

    n_workers = 36
    grid_type = "lhs"
    n_samples = 1e2  # adjust as needed

    # --------------------------------------------------
    # Anchors (absolute values to set scale)
    # --------------------------------------------------
    anchors = {
        "mu_N": 1e-4,         # [1/s]
        "D_N": 1.85,      # [µm^2/s]
        "K_A": 667,           # [concentration units]
        "K_I": 1,             # [concentration units] (tight binding for repressive interactions)
        "L_init": "constant",   # keep consistent with your ICs
        "N_sigma": 25.0,  # [concentration units] (initial Nodal pulse width)
        "sigma_N_ref": 1.0,  # [concentration units/s] (reference Nodal production rate)
        "rho_mu": 1.0,        # relative to mu_N
    }

    # --------------------------------------------------
    # Simulation config (absolute space + time)
    # --------------------------------------------------
    dx = 12.5             # µm
    L = 1500.0            # µm
    T = 10 * 60 * 60      # 10 hours in seconds

    sim_config = {
        "dx": dx,
        "L": L,
        "T": T,
        "model_class": NodalLeftyField1D,
        "tracker_class": NodalROITracker,
        "interval": 300,
    }

    # CFL-safe dt
    Dmax = anchors["D_N"] * max(1.0, np.max([10.0]))  # crude upper bound
    sim_config["dt"] = .25#0.5 * dx**2 / anchors["D_N"] / 5

    mu_N = anchors["mu_N"]
    K_A = anchors["K_A"]
    beta_a = anchors["sigma_N_ref"] / (mu_N * K_A)
    # --------------------------------------------------
    # ND parameter grid (swept in ND space!)
    # --------------------------------------------------
    param_grid = {
        "beta_r_ratio": np.logspace(-2, 1, 10),     # Lefty strength
        "kappa_NL": np.logspace(-2, 2, 10),   # Repressor threshold
        "delta": np.logspace(0, 2, 10),      # Relative diffusivity
        "a_amp": np.logspace(-2, 0, 10),      # Initial Nodal pulse
        "r_value": np.logspace(-2, 0, 10),    # Initial Lefty baseline
    }

    param_dicts = make_param_dicts(param_grid, grid_type=grid_type, n_samples=n_samples, seed=42)

    # Convert ratio → β_r before nd_to_dim(...)
    for s in param_dicts:
        if "beta_r_ratio" in s:
            s["beta_r"] = float(s["beta_r_ratio"]) * beta_a
            del s["beta_r_ratio"]

    # Convert ND → dimension-full for each run
    args = [(nd_to_dim(p, anchors), sim_config, output_dir) for p in param_dicts]

    with mp.Pool(processes=n_workers) as pool:
        for _ in tqdm(pool.imap(run_and_save_single_simulation, args), total=len(args), desc="Simulations"):
            pass

    print("Sweep complete. Results saved to:", output_dir)
