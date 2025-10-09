from symmetry_breaking_JAX.models_2D.param_classes import Params2D, BlipSet2D
from symmetry_breaking_JAX.models_2D.run import build_initial_state_2d, run_2d
import jax.numpy as jnp



def single_run_2D(dim_p, grid, T, n_save_points=121):

    X, Y = grid.X, grid.Y
    Lx = X.max() - X.min()
    Ly = Y.max() - Y.min()
    dx = jnp.abs(X[0, 0] - X[0, 1])

    # Build Params1D dataclass from dim_p dict
    p = Params2D(
        D_N=dim_p["D_N"], D_L=dim_p["D_L"],
        sigma_N=dim_p["sigma_N"], sigma_L=dim_p["sigma_L"],
        mu_N=dim_p["mu_N"], mu_L=dim_p["mu_L"],
        n=2, p=2, alpha=1.0,
        K_A=dim_p["K_A"], K_NL=dim_p["K_NL"], K_I=dim_p["K_I"],
        Lx=Lx, Ly=Ly, bc=dim_p["bc"], dx=float(dx), dy=float(dx),
        tau_impulse=60.0,
        sigma_t_direct=0.1, geometry=dim_p["geometry"]
    )

    # No stochastic blips for now
    blips = BlipSet2D.empty()

    # Initial condition
    y0 = build_initial_state_2d(
        grid,
        N_mode="gaussian", N_positions=dim_p["N_positions"], N_sigmas=dim_p["N_sigmas"], N_amps=dim_p["N_amps"],
        L_mode="constant", L_amp=dim_p["L_amp"],
        rho_value=0.1
    )

    # Run simulation
    save_ts = jnp.linspace(0, T, n_save_points)  # every ~5 min
    ts, ys = run_2d(p, blips, T=T, grid=grid, save_ts=save_ts, y0=y0)

    # unpack grid
    X, Y, = grid.X, grid.Y
    nx, ny = X.shape
    nxy = nx * ny

    # extract Nodal field over time
    N_t = ys[:, :nxy].reshape(len(ts), nx, ny)
    L_t = ys[:, nxy:2 * nxy]  # shape (nt, nxy)
    L_t = L_t.reshape(len(ts), *grid.X.shape)

    return N_t, L_t