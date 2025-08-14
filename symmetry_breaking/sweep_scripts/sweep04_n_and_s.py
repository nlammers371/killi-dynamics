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
from symmetry_breaking.models.sweep import make_param_dicts, ensure_output_dir, run_and_save_single_simulation

# Setup global vars for multiprocessing
lock = Lock()
output_file = "sweep_results.csv"


if __name__ == "__main__":
    import multiprocessing as mp
    from pathlib import Path
    sweep_name = "sweep04_n_and_s"
    root = Path("/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/symmetry_breaking/pde/sweeps/")
    # Settings
    # use_random = True  # or False for full grid
    n_workers = 28 #np.max([1, int(mp.cpu_count() / 2.1)])
    grid_type = "lhs" #"random" if use_random else "grid"
    n_samples = 1000

    # hyperparams
    dx = 10
    L = 3500
    T = 10 * 60 * 60
    # dt = 0.5 * dx ** 2 /60 / 1.25 # Stability condition for diffusion

    param_grid = {
        "K_I": np.logspace(1, 1e6, 10),
        "D_L": np.logspace(np.log10(1.85*2), np.log10(60), 10),
        "sigma_N": np.logspace(-4, 2, 10),
    }

    static_params = {
        "N_amp": 100,
        'K_NL': 138.9495494373,
        'K_A': 193.0697728883,
        # 'K_R': 19.3069772888,
        # 'K_I': 19.3069772888,
        'mu_L': 0.0002275846,
        'N_sigma': 30,
        'sigma_N': 10.0,
        'sigma_L': 10.0,
        'D_N': 2*1.85,
        # 'D_L': 15.0,
        'n': 2,
        'm': 1,
        'p': 2,
        'q': 2,
        "L_init": "constant"
    }

    sim_config = {
        "dx": dx,
        "L": L,
        "T": T,
        "model_class": NodalLeftyField1D,
        "tracker_class": NodalROITracker,
        "interval": 300,
    }

    # --- compute CFL-safe dt once (using max D0) ---
    # dx = 10
    # Dmax = max(static_params["D_N"], static_params["D_L"])
    # dt = min(sim_config.get("dt", 1e9), dt_cfl(dx, Dmax, safety=0.5))  # keep your old dt if smaller
    sim_config["dt"] = 0.5 * dx ** 2 / 60 / 1.25

    param_dicts = make_param_dicts(param_grid, grid_type=grid_type, n_samples=n_samples, seed=42)

    output_dir = ensure_output_dir(root /  sweep_name )

    print(f"Running {len(param_dicts)} simulations...")

    args = [(p | static_params, sim_config, output_dir) for p in param_dicts]

    # run_and_save_single_simulation(args[0])
    with mp.Pool(processes=n_workers) as pool:
        # Use tqdm to wrap pool.map and show progress
        for _ in tqdm(pool.imap(run_and_save_single_simulation, args), total=len(args), desc="Simulations"):
            pass  # This keeps tqdm working, but you don't need to do anything here


    print("Sweep complete. Results saved to:", output_dir)
