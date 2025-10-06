import jax
import jax.numpy as jnp
import diffrax as dfx
import numpy as np
from symmetry_breaking_JAX.models_2D.JAX_NL_2D import build_rhs_2d
from symmetry_breaking_JAX.models_2D.geom_utils import make_grid, Grid2D, SphereGrid
from symmetry_breaking_JAX.models_2D.param_classes import Params2D, BlipSet2D
from symmetry_breaking_JAX.models_1D.JAX_NL_1D import init_gaussian, init_constant, init_multi_gaussian

# ----------------------------
# Initialization helpers (2D)
# ----------------------------

def init_gaussian_2d(X, Y, amp: float, sigma: float, x0=0.0, y0=0.0):
    return amp * jnp.exp(-0.5 * (((X-x0)**2 + (Y-y0)**2) / (sigma**2)))

def init_multi_gaussian_2d(X, Y, amp: float, sigma: float, n_spots: int, key):
    # Sample random centers uniformly within bounding box
    x0 = jax.random.uniform(key, shape=(n_spots,), minval=X.min(), maxval=X.max())
    y0 = jax.random.uniform(key, shape=(n_spots,), minval=Y.min(), maxval=Y.max())
    gaussians = [
        init_gaussian_2d(X, Y, amp, sigma, cx, cy) for cx, cy in zip(x0, y0)
    ]
    return jnp.sum(jnp.stack(gaussians), axis=0)

def init_gaussian_sphere(Theta, Phi, amp: float, sigma_abs: float, R: float, center=(0.0,0.0)):
    """Gaussian on a sphere; center=(theta0,phi0)."""
    theta0, phi0 = center
    # great-circle distance
    cos_d = (
        jnp.sin(Theta)*jnp.sin(theta0)*jnp.cos(Phi-phi0)
        + jnp.cos(Theta)*jnp.cos(theta0)
    )
    dist_ang = jnp.arccos(jnp.clip(cos_d, -1.0, 1.0))
    dist_abs = R * dist_ang
    return amp * jnp.exp(-0.5 * (dist_abs/sigma_abs)**2)

def build_initial_state_2d(
    params,
    grid,
    *,
    N_positions: jnp.ndarray | None = None,  # (n_spots, 2)
    N_amps: jnp.ndarray | None = None,       # (n_spots,)
    N_sigmas: jnp.ndarray | None = None,     # (n_spots,)
    N_mode: str = "gaussian",                # "gaussian" uses spots; "constant" or "none" override it
    N_amp_default: float = 0.0,
    N_sigma_default: float = 10.0,
    n_N_spots: int = 0,                      # ignored if N_positions given
    N_key=None,
    L_mode: str = "constant",
    L_amp: float = 0.0,
    rho_value: float = 0.1,
):
    """
    Universal 2D initializer.

    - If N_positions is provided → use it directly (one Gaussian per row)
    - Else, randomly sample n_N_spots inside disk
    - Lefty: constant or zero
    """
    X, Y = grid.X, grid.Y

    # -- Handle random placement if needed --
    if N_positions is None and n_N_spots > 0:
        if N_key is None:
            N_key = jax.random.PRNGKey(np.random.randint(0, 1e9))
        key_r, key_theta = jax.random.split(N_key)
        r = jnp.sqrt(jax.random.uniform(key_r, shape=(n_N_spots,))) * (
            min(params.Lx, params.Ly) / 2.0 - 2.0 * params.dx
        )
        theta = 2 * jnp.pi * jax.random.uniform(key_theta, shape=(n_N_spots,))
        N_positions = jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta)], axis=1)
        N_sigmas = jnp.ones(n_N_spots) * N_sigma_default if N_sigmas is None else N_sigmas
        N_amps = jnp.ones(n_N_spots) * N_amp_default if N_amps is None else N_amps

    elif N_positions is None:
        N_positions = jnp.zeros((0, 2))
        N_sigmas = jnp.zeros((0,))
        N_amps = jnp.zeros((0,))

    # -- Build fields --
    if N_mode == "none":
        N0 = jnp.zeros_like(X)
    elif N_mode == "constant":
        N0 = jnp.full_like(X, N_amp_default)
    elif N_mode == "gaussian":
        def add_spot(accum, args):
            x0, y0, amp, sigma = args
            bump = amp * jnp.exp(-0.5 * ((X - x0) ** 2 + (Y - y0) ** 2) / (sigma ** 2))
            return accum + bump

        args = jnp.column_stack([N_positions[:, 0], N_positions[:, 1], N_amps, N_sigmas])
        N0 = jax.lax.fori_loop(0, args.shape[0], lambda i, acc: add_spot(acc, args[i]), jnp.zeros_like(X))
    else:
        raise ValueError(f"Unknown N_mode '{N_mode}'")

    # -- Lefty constant/none --
    L0 = jnp.full_like(X, L_amp) if L_mode == "constant" else jnp.zeros_like(X)
    rho0 = jnp.full_like(X, rho_value)
    F_N0 = jnp.zeros_like(X)
    F_L0 = jnp.zeros_like(X)

    y0 = jnp.concatenate([N0.ravel(), L0.ravel(), rho0.ravel(), F_N0.ravel(), F_L0.ravel()])
    return y0


# ----------------------------
# Runner
# ----------------------------
def run_2d(
    params: Params2D,
    blips: BlipSet2D,
    grid: Grid2D | SphereGrid,
    T: float,
    save_ts: jnp.ndarray,
    y0: jnp.ndarray | None = None,
):
    """Integrate the 2-D Nodal–Lefty PDE on a provided grid."""

    rhs = build_rhs_2d(params, blips, grid)

    if y0 is None:
        raise ValueError("run_2d() now requires y0_override; grid initialization is external.")

    term = dfx.ODETerm(rhs)
    solver = dfx.Dopri5()
    controller = dfx.PIDController(rtol=1e-4, atol=1e-6)

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=T,
        dt0=1e-3,
        y0=y0,
        args=None,
        saveat=dfx.SaveAt(ts=save_ts),
        stepsize_controller=controller,
        max_steps=10_000_000,
    )
    return sol.ts, sol.ys


# def simulate_2d(params, blips, T, nx=101, ny=101, save_ts=None, **init_kwargs):
#     grid = make_grid(params, nx, ny)
#     y0 = build_initial_state_2d(params, grid, **init_kwargs)
#     return run_2d(
#         params,
#         blips,
#         grid,
#         T=T,
#         save_ts=save_ts or jnp.linspace(0, T, 200),
#         y0=y0,
#     )

