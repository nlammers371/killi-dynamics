import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Lock, Manager
import os
import time
import uuid
from symmetry_breaking.models.NLP_neutralization_1D import NodalLeftyNeutralization1D
from symmetry_breaking.models.trackers import NodalROITracker
from symmetry_breaking.models.sweep import make_param_dicts, ensure_output_dir, run_and_save_single_simulation

# Setup global vars for multiprocessing
lock = Lock()
output_file = "sweep_results.csv"


if __name__ == "__main__":
    import multiprocessing as mp
    from pathlib import Path
    sweep_name = "sweep01_neutralization_v2"
    root = Path("/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/symmetry_breaking/pde/sweeps/")
    # Settings
    use_random = True  # or False for full grid
    n_workers = 12 #np.max([1, int(mp.cpu_count() / 2.1)])
    grid_type = "lhs" #"random" if use_random else "grid"
    n_samples = 100000

    # hyperparams
    dx = 10
    L = 3000
    T = 10 * 60 * 60
    # dt = 0.5 * dx ** 2 /60 / 1.25 # Stability condition for diffusion

    param_grid = {
        # "sigma_L": np.logspace(-2, 0, 10),
        # "sigma_N": np.logspace(0, 2, 10),
        "K_NL": np.logspace(1, 3, 8),
        "K_A": np.logspace(2, 4, 8),
        "K_R": np.logspace(1, 3, 8),
        "sigma_L": np.logspace(-1, 1, 8),
        # "K_rho": np.logspace(1.81, 3.81, 10),
        "K_I": np.logspace(1, 3, 8),
        # "alpha_N": np.linspace(0.46, 5, 8),
        "mu_L": np.logspace(-4.3, -2, 8),
        "N_amp": np.logspace(1, 4, 8),
        "N_sigma": np.logspace(1, 2, 3),
    }

    static_params = {
                      "sigma_N": 10.0,  # Nodal auto-activation
                      # "sigma_L": 0.1,  # Lefty production
                      "D0_N": 1.85,
                      "D0_L": 15.0,
                      "no_density_dependence": True,
                      # "mu_N": 1.11e-4,
                      # "mu_L": 0.61e-4,
                      "alpha_L": 0,
                      "alpha_N": 0,
                      "tau_rho": 3600,
                      "n": 2,
                      "m": 1,
                      "p": 2,
                      "q": 2,
                      # "N_amp":5000,
                      # "N_sigma":50.0,
                    }

    # Simulation config (you can also pass this from CLI)
    sim_config = {
                    "dx": dx,
                    "L": L,
                    "T": T,
                    "model_class": NodalLeftyNeutralization1D,
                    "tracker_class": NodalROITracker,
                    "interval": 1000,
                }

    # --- compute CFL-safe dt once (using max D0) ---
    # dx = 10
    Dmax = max(static_params["D0_N"], static_params["D0_L"])
    # dt = min(sim_config.get("dt", 1e9), dt_cfl(dx, Dmax, safety=0.5))  # keep your old dt if smaller
    sim_config["dt"] = 0.5 * dx ** 2 / Dmax / 1.25

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
