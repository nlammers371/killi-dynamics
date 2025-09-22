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
    sweep_name = "sweep07_field_tuning"
    root = Path("/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/symmetry_breaking/pde/sweeps/")
    # Settings
    # use_random = True  # or False for full grid
    n_workers = 36 #np.max([1, int(mp.cpu_count() / 2.1)])
    grid_type = "lhs" #"random" if use_random else "grid"
    n_samples = 1000

    # hyperparams
    dx = 10
    L = 3500
    T = 10 * 60 * 60
    # dt = 0.5 * dx ** 2 /60 / 1.25 # Stability condition for diffusion

    param_grid = {
        'rate_N': np.logspace(np.log10(1/T), np.log10(1/60), 10) / L,
        "D_L": np.linspace(15, 60, 10),
        "L_value": np.linspace(100, 150, 10)
    }

    static_params = {
        'K_NL': 667,  # From lit
        'K_A': 667,  # From lit
        'K_I': 10,  # not in lit model
        "N_amp":0,
        'mu_L': 0.0002275846,  # 5x larger than lit
        'N_sigma': 31.6227766017,  # NA
        'sigma_N': 10,  # 10x lit
        'sigma_L': 10,  # 1000x (!!) lit
        'D_ratio': 1.85/15.0,
        # 'D_N': 2 * 1.85,  # 2X lit
        # 'D_L': 2 * 15.0,  # 2x lit
        'n': 2,  # lit
        'm': 1,  # NA
        'p': 2,  # lit
        'q': 2,
        "L_init": "constant",
        'stoch_to_N': True,
        # 'rate_N': 1 / (3500 * 60) / 3,
        'amp_median_N': 500,
        'amp_sigma_N': .00001,
        "sigma_x": 31}

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
    Dmax = 60
    # dt = min(sim_config.get("dt", 1e9), dt_cfl(dx, Dmax, safety=0.5))  # keep your old dt if smaller
    sim_config["dt"] = 0.5 * dx ** 2 / Dmax / 1.25

    param_dicts = make_param_dicts(param_grid, grid_type=grid_type, n_samples=n_samples, seed=42)

    output_dir = ensure_output_dir(root /  sweep_name )

    print(f"Running {len(param_dicts)} simulations...")

    args_orig = [(p | static_params, sim_config, output_dir) for p in param_dicts]
    args = []
    for a, arg in enumerate(args_orig):
        sim_config = arg[1].copy()
        param_dict = arg[0].copy()
        sim_config["dt"] = 0.5 * dx ** 2 / param_dict["D_L"] / 1.25
        # arg[1] = sim_config
        args.append((arg[0], sim_config, arg[2]))

    # run_and_save_single_simulation(args[0])
    with mp.Pool(processes=n_workers) as pool:
        # Use tqdm to wrap pool.map and show progress
        for _ in tqdm(pool.imap(run_and_save_single_simulation, args), total=len(args), desc="Simulations"):
            pass  # This keeps tqdm working, but you don't need to do anything here


    print("Sweep complete. Results saved to:", output_dir)
