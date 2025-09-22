import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Lock, Manager
import os
import time
import uuid
from symmetry_breaking.models.NL_RD_1D import TuringPDE1D
from symmetry_breaking.models.sweep import run_simulation_1D, make_1d_grid
from symmetry_breaking.models.trackers import NodalROITracker

# Setup global vars for multiprocessing
lock = Lock()
output_file = "sweep_results.csv"

def run_and_save_single_simulation(args):

    param_dict, sim_config, output_dir = args

    # Unpack sim config
    dx = sim_config["dx"]
    L = sim_config["L"]
    T = sim_config["T"]
    dt = sim_config["dt"]
    model_class = sim_config["model_class"]
    tracker_class = sim_config["tracker_class"]
    interval = sim_config.get("interval", 1000)

    # Generate 1D grid
    grid = make_1d_grid(length=L, dx=dx)

    # Run sim
    result = run_simulation_1D(param_dict=param_dict,
                               grid=grid,
                               model_class=model_class,
                               tracker_class=tracker_class,
                               dt=dt,
                               T=T,
                               interval=interval)

    # Save results
    run_id = str(uuid.uuid4())[:8]
    output_data = {**param_dict, **result}
    output_path = os.path.join(output_dir, f"result_{run_id}.json")
    pd.Series(output_data).to_json(output_path)

# def generate_param_dicts(param_grid):
#     keys = list(param_grid.keys())
#     value_lists = [param_grid[k] for k in keys]
#     all_combos = list(itertools.product(*value_lists))
#     return [dict(zip(keys, combo)) for combo in all_combos]

def ensure_output_dir(path: str | os.PathLike):
    os.makedirs(path, exist_ok=True)
    return path

def make_param_dicts(param_grid, grid_type="grid", n_samples=1000, seed=42):
    """
    Generate parameter dictionaries.
    """

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    all_combinations = list(itertools.product(*value_lists))

    if grid_type == "random":
        np.random.seed(seed)
        sampled = np.random.choice(len(all_combinations), size=n_samples, replace=False)
        return [dict(zip(keys, all_combinations[i])) for i in sampled]
    else:
        return [dict(zip(keys, combo)) for combo in all_combinations]

if __name__ == "__main__":
    import multiprocessing as mp
    from pathlib import Path
    sweep_name = "sweep01"
    root = Path("/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/symmetry_breaking/pde/sweeps/")
    # Settings
    use_random = True  # or False for full grid
    n_workers = np.max([1, int(mp.cpu_count() /1.25)])
    grid_type = "random" if use_random else "grid"
    n_samples = 100000

    # hyperparams
    dx = 10
    L = 3500
    T = 10 * 60 * 60
    dt = 0.5 * dx ** 2 / 15 / 1.25 # Stability condition for diffusion

    param_grid = {
        "V_R": np.logspace(-3, -1, 10),
        "V_A": np.logspace(-1, 1, 10),
        "lambda_": np.logspace(-2, 0, 10),
        "K_R": np.logspace(1, 3, 10),
        "K_A": np.logspace(1.81, 3.81, 10),
        "K_P": np.logspace(1.81, 3.81, 10),
    }

    static_params = {
                  "da": 1.85,
                  "dr": 15,
                  "k_a": 1.11e-4,
                  "k_r": 0.61e-4,
                  "V_A": 1.0,
                  "V_R": 1e-2,
                  "lambda_": 0.01,    # Lefty inhibition rate
                  "n": 2,
                  "m": 2,
                  "p": 2
                    }

    # Simulation config (you can also pass this from CLI)
    sim_config = {
                    "dx": dx,
                    "L": L,
                    "T": T,
                    "dt": dt,
                    "model_class": TuringPDE1D,
                    "tracker_class": NodalROITracker,
                    "interval": 1000,
                }

    param_dicts = make_param_dicts(param_grid, grid_type=grid_type, n_samples=n_samples)
    output_dir = ensure_output_dir(root /  sweep_name )

    print(f"Running {len(param_dicts)} simulations...")

    args = [(p | static_params, sim_config, output_dir) for p in param_dicts]

    with mp.Pool(processes=n_workers) as pool:
        # Use tqdm to wrap pool.map and show progress
        for _ in tqdm(pool.imap(run_and_save_single_simulation, args), total=len(args), desc="Simulations"):
            pass  # This keeps tqdm working, but you don't need to do anything here

    print("Sweep complete. Results saved to:", output_dir)
