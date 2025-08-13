from joblib import Parallel, delayed
from pde import ScalarField, FieldCollection, PDEBase, CartesianGrid
import os
import time
import uuid
import numpy as np
import pandas as pd

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

def make_1d_grid(length=3000, dx=10, periodic=True):
    N = int(length / dx)
    grid = CartesianGrid([[0, length]], shape=(N,), periodic=periodic)
    return grid


def run_simulation_1D(param_dict, grid, model_class, tracker_class, dt=10, T=36000, interval=100):

    adjusted_interval = int(interval / dt )
    model = model_class(**param_dict)
    state = model.get_state(grid)

    tracker = tracker_class(grid, interval=adjusted_interval)
    _ = model.solve(state, t_range=T, dt=dt, tracker=tracker)

    # Merge input parameters and tracked metrics
    result = {
        **param_dict,
        **tracker.get_metrics(),
        **tracker.get_profiles(),
    }

    return result
#
# results = Parallel(n_jobs=16)(
#     delayed(run_simulation)(param) for param in param_list
# )