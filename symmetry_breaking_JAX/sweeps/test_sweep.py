# sweep_jax.py
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from pathlib import Path
from symmetry_breaking_JAX.helpers import single_run, split_numeric_and_static, make_param_dicts, ensure_output_dir
from symmetry_breaking.utilities.helpers import nd_to_dim
import json

# --------------------------------------------------
# Sweep script
# --------------------------------------------------
if __name__ == "__main__":

    sweep_name = "sweep01_jax_stable"
    root = Path("/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/symmetry_breaking/pde/sweeps/")
    # output_dir = ensure_output_dir(root / sweep_name)

    # Grid + sample settings
    grid_type = "lhs"
    n_samples = 10000  # adjust as needed

    # --------------------------------------------------
    # Anchors (absolute values to set scale)
    # --------------------------------------------------
    anchors = {
        "mu_N": 1e-4,         # [1/s]
        "D_N": 1.85,          # [µm^2/s]
        "K_A": 667,           # [concentration units]
        "K_I": 1,             # [concentration units]
        "L_init": "constant",
        "N_sigma": 25.0,
        "sigma_N_ref": 1.0,   # [conc units/s]
        "rho_mu": 1.0,
    }

    # --------------------------------------------------
    # Simulation config
    # --------------------------------------------------
    dx = 12.5            # µm
    L = 1500.0           # µm
    T = 10 * 60 * 60     # 10 h
    n_save_points = 121  # save every ~5 min
    nx = int(L / dx) + 1
    x = jnp.linspace(0, L, nx)

    # CFL-safe-ish dt (not used by Diffrax if adaptive, but keep for reference)
    sim_dt = 0.25

    mu_N = anchors["mu_N"]
    K_A = anchors["K_A"]
    beta_a = anchors["sigma_N_ref"] / (mu_N * K_A)

    # --------------------------------------------------
    # ND parameter grid (swept in ND space)
    # --------------------------------------------------
    param_grid = {
        "beta_r_ratio": np.logspace(-2, 1, 10),
        "kappa_NL": np.logspace(-2, 2, 10),
        "delta": np.logspace(0, 2, 10),
        "a_amp": np.logspace(-2, 0, 10),
        "r_value": np.logspace(-2, 0, 10),
    }
    param_dicts = make_param_dicts(param_grid, grid_type=grid_type, n_samples=n_samples, seed=42)

    # Convert ratio → β_r
    for s in param_dicts:
        s["beta_r"] = float(s["beta_r_ratio"]) * beta_a
        del s["beta_r_ratio"]

    # Convert ND → dimension-full
    dim_params = [nd_to_dim(p, anchors) for p in param_dicts]

    # Vectorize across parameter sets
    batched_run = jax.vmap(single_run, in_axes=(0, None, None, None, None, None, None))

    # --------------------------------------------------
    # Run sweep on GPU
    # --------------------------------------------------
    # Split into numeric vs static
    numeric_params, static_params = split_numeric_and_static(dim_params)
    # dim_params_jax = jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *numeric_params)

    batch_size = 128
    n = len(param_dicts)

    batched_run = jax.jit(jax.vmap(single_run, in_axes=(0, None, None, None, None, None, None)),
                          static_argnums=(2, 3, 4, 5, 6) )

    # Warm-up: trigger compilation with one small batch
    print("Warming up JAX compilation...")
    dummy_batch = {k: v[:1] for k, v in numeric_params.items()}
    _ = batched_run(dummy_batch, x, dx, L, T, nx, n_save_points)  # compilation happens here

    # Now time the real runs
    N_results = []
    L_results = []
    for i in tqdm(range(0, n, batch_size), desc="Running sweeps..."):
        batch = {k: v[i:i + batch_size] for k, v in numeric_params.items()}
        out = batched_run(batch, x, dx, L, T, nx, n_save_points)
        N_results.append(out[0])
        L_results.append(out[1])

    N_results = jnp.concatenate(N_results)
    L_results = jnp.concatenate(L_results)

    # Convert param_dicts to JSON-friendly (floats, lists)
    with open(output_dir / "params.json", "w") as f:
        json.dump(param_dicts, f, indent=2)

    # Save Nodal and Lefty arrays as compressed npz
    np.savez_compressed(
        output_dir / "results.npz",
        N=N_results,  # shape: (n_samples, n_timepoints, nx)
        L=L_results,  # shape: (n_samples, n_timepoints, nx)
    )
    print("Sweep complete. Saved results + params to:", output_dir)

    # Save results
    # np.save(output_dir / "results.npy", np.array(N_results))
    # print("Sweep complete. Results saved to:", output_dir)
