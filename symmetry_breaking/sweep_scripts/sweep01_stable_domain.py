import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Lock, Manager
import os
import time
import uuid
from symmetry_breaking.models.NL_field_1D import NodalLeftyField1D
from symmetry_breaking.models.trackers import NodalROITracker
from symmetry_breaking.utilities.helpers import nd_to_dim
from symmetry_breaking.models.sweep import make_param_dicts, ensure_output_dir, run_and_save_single_simulation

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
# Conversion: ND → dimension-full
# --------------------------------------------------
def nd_to_dim(nd_params, anchors):
    """
    Convert ND params to dimension-full params for NodalLeftyField1D.

    anchors: must include
        mu_N : float  [1/s]
        D_N  : float  [µm^2/s]
        K_A  : float  [concentration units]
    """
    mu_N = anchors["mu_N"]
    D_N = anchors["D_N"]
    K_A = anchors["K_A"]

    out = {}

    # Production rates
    out["sigma_N"] = nd_params.get("beta_a", 1.0) * mu_N * K_A
    out["sigma_L"] = nd_params.get("beta_r", 1.0) * mu_N * K_A

    # Decay rates
    out["mu_N"] = mu_N
    out["mu_L"] = nd_params.get("rho_mu", 1.0) * mu_N

    # Diffusion
    out["D_N"] = D_N
    out["D_L"] = nd_params.get("delta", 1.0) * D_N

    # Binding constants
    out["K_A"] = K_A
    out["K_I"] = nd_params.get("kappa_I", 1e-3) * K_A  # default small κ_I
    out["K_NL"] = nd_params.get("kappa_NL", 1.0) * K_A

    # Hill exponents (optional)
    for key in ["n", "m", "p", "q"]:
        if key in nd_params:
            out[key] = nd_params[key]

    # Initial conditions
    if "a_amp" in nd_params:
        out["N_amp"] = nd_params["a_amp"] * K_A
    if "r_value" in nd_params:
        out["L_value"] = nd_params["r_value"] * K_A

    # Pass through anchors if relevant (e.g. init modes)
    for k in ["L_init", "N_sigma"]:
        if k in anchors:
            out[k] = anchors[k]

    return out


# --------------------------------------------------
# Sweep script
# --------------------------------------------------
if __name__ == "__main__":
    sweep_name = "sweep_nd_example"
    root = Path(
        "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/symmetry_breaking/pde/sweeps/"
    )
    output_dir = ensure_output_dir(root / sweep_name)

    n_workers = 36
    grid_type = "lhs"
    n_samples = 200  # adjust as needed

    # --------------------------------------------------
    # Anchors (absolute values to set scale)
    # --------------------------------------------------
    anchors = {
        "mu_N": 1e-4,         # [1/s]
        "D_N": 1.85,      # [µm^2/s]
        "K_A": 667,           # [concentration units]
        "L_init": "constant",   # keep consistent with your ICs
    }

    # --------------------------------------------------
    # Simulation config (absolute space + time)
    # --------------------------------------------------
    dx = 10.0             # µm
    L = 2000.0            # µm
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
    sim_config["dt"] = 0.5 * dx**2 / anchors["D_N"] / 1.25

    # --------------------------------------------------
    # ND parameter grid (swept in ND space!)
    # --------------------------------------------------
    param_grid = {
        "beta_r": np.logspace(-2, 2, 10),     # Lefty strength
        "kappa_NL": np.logspace(-2, 2, 10),   # Repressor threshold
        "delta": np.logspace(0, 2, 10),      # Relative diffusivity
        "a_amp": np.logspace(-2, 2, 10),      # Initial Nodal pulse
        "r_value": np.logspace(-2, 2, 10),    # Initial Lefty baseline
    }

    param_dicts = make_param_dicts(param_grid, grid_type=grid_type, n_samples=n_samples, seed=42)

    # Convert ND → dimension-full for each run
    args = [(nd_to_dim(p, anchors), sim_config, output_dir) for p in param_dicts]

    print(f"Running {len(args)} simulations...")

    with mp.Pool(processes=n_workers) as pool:
        for _ in tqdm(pool.imap(run_and_save_single_simulation, args), total=len(args), desc="Simulations"):
            pass

    print("Sweep complete. Results saved to:", output_dir)
