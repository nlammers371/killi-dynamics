import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from symmetry_breaking_JAX.models_2D.run import build_initial_state_2d, run_2d
from symmetry_breaking_JAX.models_2D.geom_utils import make_grid
from symmetry_breaking_JAX.models_2D.helpers_2D import single_run_2D


T = 3600 * 5
n_save = 50

# 1. Define parameters and blips
params = {
    "D_N": 10.0,
    "D_L": 23.9 / 2.0,
    "sigma_N": 10.0,
    "sigma_L": 1e-6,
    "mu_N": 0.0001,
    "mu_L": 0.0001,
    "n": 2,
    "p": 2,
    "alpha": 1.0,
    "K_A": 667.0,
    "K_NL": 1e6,
    "K_I": 1e6,
    "Lx": 1250.0,
    "Ly": 100.0,
    "dx": 10.0,
    "dy": 10.0,
    "geometry": "rectangle",
    "bc": "periodic",
    "N_positions": jnp.array([[0, 0]]),
    "N_amps": jnp.array([1000]),
    "N_sigmas": jnp.array([25]),
    "L_mode": "constant",
    "L_amp": 0.0
}

grid = make_grid(params)

N_t, L_t = single_run_2D(params, grid, T)