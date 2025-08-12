import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Lock, Manager
import os
import time
import uuid
from symmetry_breaking.models.NLP_neutralization_1D import NodalLeftyNeutralization1D
from symmetry_breaking.models.sweep import run_simulation_1D, make_1d_grid
from symmetry_breaking.models.trackers import NodalROITracker
from symmetry_breaking.utilities.parameter_definitions import dt_cfl

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


def ensure_output_dir(path: str | os.PathLike):
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
            idx = _lhs_indices(len(v), n, rng)   # stratified indices for this dim
            idx_matrix.append(idx)
        idx_matrix = np.stack(idx_matrix, axis=1)  # shape (n_samples, n_params)
        # Build param dicts by taking the i-th pick from each dimension
        out = []
        for row in idx_matrix:
            out.append({k: values[j][row[j]].item() for j, k in enumerate(keys)})
        return out

    raise ValueError(f"Unknown grid_type: {grid_type}")

if __name__ == "__main__":
    import multiprocessing as mp
    from pathlib import Path
    sweep_name = "sweep02_neutralization"
    root = Path("/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/symmetry_breaking/pde/sweeps/")
    # Settings
    use_random = True  # or False for full grid
    n_workers = np.max([1, int(mp.cpu_count() /1.25)])
    grid_type = "lhs" #"random" if use_random else "grid"
    n_samples = 100000

    # hyperparams
    dx = 10
    L = 3500
    T = 1 * 60 * 60
    # dt = 0.5 * dx ** 2 /60 / 1.25 # Stability condition for diffusion

    param_grid = {
        # "sigma_L": np.logspace(-2, 0, 10),
        # "sigma_N": np.logspace(0, 2, 10),
        "K_NL": np.logspace(1, 3, 10),
        "K_A": np.logspace(2, 4, 10),
        "K_R": np.logspace(1, 3, 10),
        # "K_rho": np.logspace(1.81, 3.81, 10),
        "K_I": np.logspace(1, 3, 10),
        "alpha_N": np.linspace(0.46, 5, 10),
        "mu_L": np.logspace(-4.3, -2, 10),
        "N_amp": np.logspace(1, 4, 10),
        "N_sigma": np.logspace(1, 2, 3),
    }

    static_params = {
                      "sigma_N": 10.0,  # Nodal auto-activation
                      "sigma_L": 0.1,  # Lefty production
                      "D0_N": 60.0,
                      "D0_L": 60.0,
                      # "mu_N": 1.11e-4,
                      # "mu_L": 0.61e-4,
                      "alpha_L": 0.46,
                      "tau_rho": 3600,
                      "n": 2,
                      "m": 2,
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

    args = [(p | static_params, sim_config, "output_dir") for p in param_dicts]

    # run_and_save_single_simulation(args[0])
    with mp.Pool(processes=n_workers) as pool:
        # Use tqdm to wrap pool.map and show progress
        for _ in tqdm(pool.imap(run_and_save_single_simulation, args), total=len(args), desc="Simulations"):
            pass  # This keeps tqdm working, but you don't need to do anything here


    print("Sweep complete. Results saved to:", output_dir)
