import jax
import jax.numpy as jnp
import diffrax as dfx

from symmetry_breaking_JAX.models_2D.JAX_NL_2D import build_rhs_2d
from symmetry_breaking_JAX.models_2D.param_classes import Params2D, BlipSet2D
from symmetry_breaking_JAX.models_1D.JAX_NL_1D import init_gaussian, init_constant, init_multi_gaussian

# ----------------------------
# Initialization helpers (2D)
# ----------------------------

def init_gaussian_2d(X, Y, amp: float, sigma: float, x0=0.0, y0=0.0):
    return amp * jnp.exp(-0.5 * (((X-x0)**2 + (Y-y0)**2) / (sigma**2)))

def init_multi_gaussian_2d(X, Y, amp: float, sigma: float, n_spots: int, key):
    nx, ny = X.shape
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

def build_initial_state_2d(params: Params2D,
                           grid,
                           *,
                           N_mode="gaussian", N_amp=10.0, N_sigma=25.0, n_N_spots=1, N_key=None,
                           L_mode="constant", L_amp=0.0, n_L_spots=1, L_key=None,
                           rho_value=0.1):
    geom = params.geometry
    if geom == "disk":
        X, Y, mask = grid
        if N_mode == "gaussian" and n_N_spots == 1:
            N0 = init_gaussian_2d(X, Y, N_amp, N_sigma)
        elif N_mode == "gaussian" and n_N_spots > 1:
            if N_key is None:
                N_key = jax.random.PRNGKey(0)
            N0 = init_multi_gaussian_2d(X, Y, N_amp, N_sigma, n_N_spots, N_key)
        elif N_mode == "constant":
            N0 = init_constant(X, N_amp)
        else:
            raise ValueError("Unsupported N_mode for disk")

        if L_mode == "gaussian":
            L0 = init_gaussian_2d(X, Y, L_amp, N_sigma)
        elif L_mode == "constant":
            L0 = init_constant(X, L_amp)
        else:
            raise ValueError("Unsupported L_mode for disk")

    elif geom == "sphere":
        Theta, Phi = grid
        if N_mode == "gaussian":
            N0 = init_gaussian_sphere(Theta, Phi, N_amp, N_sigma, params.R, center=(0.0,0.0))
        elif N_mode == "constant":
            N0 = init_constant(Theta, N_amp)
        else:
            raise ValueError("Unsupported N_mode for sphere")

        if L_mode == "gaussian":
            L0 = init_gaussian_sphere(Theta, Phi, L_amp, N_sigma, params.R, center=(0.0,0.0))
        elif L_mode == "constant":
            L0 = init_constant(Theta, L_amp)
        else:
            raise ValueError("Unsupported L_mode for sphere")

    rho0 = init_constant(N0, rho_value)
    F_N0 = jnp.zeros_like(N0)
    F_L0 = jnp.zeros_like(N0)

    y0 = jnp.concatenate([N0.ravel(), L0.ravel(), rho0.ravel(),
                          F_N0.ravel(), F_L0.ravel()])
    return y0

# ----------------------------
# Runner
# ----------------------------

def run_2d(params: Params2D,
           blips: BlipSet2D,
           T: float,
           save_ts: jnp.ndarray,
           y0_override: jnp.ndarray | None = None):

    geom = params.geometry

    # --- Build grid ---
    if geom == "disk":
        nx = int(params.Lx / params.dx) + 1
        ny = int(params.Ly / params.dy) + 1
        x = jnp.linspace(-params.Lx/2, params.Lx/2, nx)
        y = jnp.linspace(-params.Ly/2, params.Ly/2, ny)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        R = jnp.sqrt(X**2 + Y**2)
        mask = R <= min(params.Lx, params.Ly)/2
        grid = (X, Y, mask)

    elif geom == "sphere":
        thetas = jnp.linspace(0.0, jnp.pi, params.n_theta)
        phis   = jnp.linspace(0.0, 2.0*jnp.pi, params.n_phi, endpoint=False)
        Theta, Phi = jnp.meshgrid(thetas, phis, indexing="ij")
        grid = (Theta, Phi)

    else:
        raise ValueError("geometry must be 'disk' or 'sphere'")

    # --- RHS ---
    rhs = build_rhs_2d(params, blips, grid)

    # --- Initial state ---
    if y0_override is not None:
        y0 = y0_override
    else:
        y0 = build_initial_state_2d(params, grid)

    # --- Solve ---
    term = dfx.ODETerm(rhs)
    solver = dfx.Dopri5()
    controller = dfx.PIDController(rtol=1e-4, atol=1e-6)

    sol = dfx.diffeqsolve(
        term, solver,
        t0=0.0, t1=T, dt0=1e-3,
        y0=y0, args=None,
        saveat=dfx.SaveAt(ts=save_ts),
        stepsize_controller=controller,
        max_steps=10_000_000
    )

    return grid, sol.ts, sol.ys
