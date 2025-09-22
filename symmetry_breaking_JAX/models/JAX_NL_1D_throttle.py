# jax_nl1d.py
from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import diffrax as dfx
import numpy as np

# ----------------------------
# Utilities
# ----------------------------
def laplace_1d(u: jnp.ndarray, dx: float, bc: str = "neumann") -> jnp.ndarray:
    if bc == "periodic":
        return (jnp.roll(u, -1) - 2.0 * u + jnp.roll(u, +1)) / (dx * dx)
    elif bc == "neumann":
        # mirror the edge values
        uL = jnp.concatenate([u[:1], u[:-1]])
        uR = jnp.concatenate([u[1:], u[-1:]])
        return (uL - 2.0 * u + uR) / (dx * dx)
    else:
        raise ValueError("bc must be 'periodic' or 'neumann'")

def hill01(x: jnp.ndarray, n: int) -> jnp.ndarray:
    x = jnp.clip(x, 0.0, 1e6)
    xn = jnp.power(x, n)
    return xn / (1.0 + xn)

# Time-Gaussian “blip” kernel (area = amp). Use small sigma_t to approximate a direct jump.
def time_kernel(t: float, t0: jnp.ndarray, sigma_t: float) -> jnp.ndarray:
    # normalized so integral over t is 1
    st = jnp.maximum(sigma_t, 1e-12)
    return jnp.exp(-0.5 * ((t - t0) / st) ** 2) / (jnp.sqrt(2.0 * jnp.pi) * st)

def space_kernel_1d(x: jnp.ndarray, x0: jnp.ndarray, sigma_x: jnp.ndarray) -> jnp.ndarray:
    sx = jnp.maximum(sigma_x, 1e-12)
    return jnp.exp(-0.5 * ((x[:, None] - x0[None, :]) / sx[None, :]) ** 2)

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Params1D:
    # Diffusion
    D_N: float
    D_L: float
    # Kinetics
    sigma_N: float
    sigma_L: float
    mu_N: float
    mu_L: float
    # Hill / interactions
    n: int
    p: int
    alpha: float
    K_A: float
    K_NL: float
    K_I: float
    apply_to_L: bool = True
    # Geometry / numerics
    Lx: float = 1500.0
    dx: float = 10.0
    bc: str = "neumann"
    # Impulse “leak” time constant
    tau_impulse: float = 1.0
    # Time “width” to realize direct blips smoothly (seconds)
    sigma_t_direct: float = 0.05

@dataclass
class BlipSet1D:
    """
    Pre-sampled events for each channel, passed as arrays:
      times: shape (M,)
      x:     shape (M,)
      amp:   shape (M,)
      sigma_x: shape (M,)   (spatial width)
    Interpretation:
      - direct events add to S_dir via time_kernel with sigma_t_direct
      - impulse events add to F via the same time_kernel; S_imp = F / tau_impulse
    """
    # Nodal (activator) direct / impulse
    N_dir_times: jnp.ndarray
    N_dir_x: jnp.ndarray
    N_dir_amp: jnp.ndarray
    N_dir_sigx: jnp.ndarray

    N_imp_times: jnp.ndarray
    N_imp_x: jnp.ndarray
    N_imp_amp: jnp.ndarray
    N_imp_sigx: jnp.ndarray

    # Lefty (repressor)
    L_dir_times: jnp.ndarray
    L_dir_x: jnp.ndarray
    L_dir_amp: jnp.ndarray
    L_dir_sigx: jnp.ndarray

    L_imp_times: jnp.ndarray
    L_imp_x: jnp.ndarray
    L_imp_amp: jnp.ndarray
    L_imp_sigx: jnp.ndarray

    @classmethod
    def empty(cls):
        """Create an empty BlipSet1D with all fields as empty arrays."""
        return cls(
            N_dir_times=jnp.array([]),
            N_dir_x=jnp.array([]),
            N_dir_amp=jnp.array([]),
            N_dir_sigx=jnp.array([]),

            N_imp_times=jnp.array([]),
            N_imp_x=jnp.array([]),
            N_imp_amp=jnp.array([]),
            N_imp_sigx=jnp.array([]),

            L_dir_times=jnp.array([]),
            L_dir_x=jnp.array([]),
            L_dir_amp=jnp.array([]),
            L_dir_sigx=jnp.array([]),

            L_imp_times=jnp.array([]),
            L_imp_x=jnp.array([]),
            L_imp_amp=jnp.array([]),
            L_imp_sigx=jnp.array([]),
        )

# ----------------------------
# RHS builder (adds auxiliary accumulators F_N, F_L for impulse mode)
# State layout: y = [N (nx), L (nx), rho (nx), F_N (nx), F_L (nx)]
# ----------------------------
def build_rhs_1d(x: jnp.ndarray, params: Params1D, blips: BlipSet1D):
    nx = x.size
    alpha = params.alpha

    # Precompute spatial Gaussians per event (constant in time → OK to precompute & close over)
    def precompute_space_kernels(x, xs, sigs):
        if xs.size == 0:
            return jnp.zeros((nx, 0))
        return space_kernel_1d(x, xs, sigs)

    Sx_N_dir = precompute_space_kernels(x, blips.N_dir_x, blips.N_dir_sigx)
    Sx_N_imp = precompute_space_kernels(x, blips.N_imp_x, blips.N_imp_sigx)
    Sx_L_dir = precompute_space_kernels(x, blips.L_dir_x, blips.L_dir_sigx)
    Sx_L_imp = precompute_space_kernels(x, blips.L_imp_x, blips.L_imp_sigx)

    def time_source(t, times, amps, sigma_t):
        if times.size == 0:
            return jnp.zeros((times.size,))
        return amps * time_kernel(t, times, sigma_t)

    def rhs(t, y, args):
        # unpack state
        N = y[0:nx]
        L = y[nx:2*nx]
        rho = y[2*nx:3*nx]
        F_N = y[3*nx:4*nx]
        F_L = y[4*nx:5*nx]

        # diffusion
        dN = params.D_N * laplace_1d(N, params.dx, params.bc)
        dL = params.D_L * laplace_1d(L, params.dx, params.bc)
        drho = jnp.zeros_like(rho)  # passive, as in your class

        # sources: direct (time-Gaussian; area = amplitude)
        S_N_dir_t = time_source(t, blips.N_dir_times, blips.N_dir_amp, params.sigma_t_direct)  # (Mn_dir,)
        S_L_dir_t = time_source(t, blips.L_dir_times, blips.L_dir_amp, params.sigma_t_direct)

        S_N_dir = (Sx_N_dir @ S_N_dir_t) if Sx_N_dir.shape[1] > 0 else jnp.zeros(nx)
        S_L_dir = (Sx_L_dir @ S_L_dir_t) if Sx_L_dir.shape[1] > 0 else jnp.zeros(nx)

        # impulse accumulators: dF/dt = input(t) - F/tau
        S_N_imp_t = time_source(t, blips.N_imp_times, blips.N_imp_amp, params.sigma_t_direct)
        S_L_imp_t = time_source(t, blips.L_imp_times, blips.L_imp_amp, params.sigma_t_direct)
        inp_N = (Sx_N_imp @ S_N_imp_t) if Sx_N_imp.shape[1] > 0 else jnp.zeros(nx)
        inp_L = (Sx_L_imp @ S_L_imp_t) if Sx_L_imp.shape[1] > 0 else jnp.zeros(nx)

        dF_N = inp_N - F_N / jnp.maximum(params.tau_impulse, 1e-12)
        dF_L = inp_L - F_L / jnp.maximum(params.tau_impulse, 1e-12)
        S_N_imp = F_N / jnp.maximum(params.tau_impulse, 1e-12)
        S_L_imp = F_L / jnp.maximum(params.tau_impulse, 1e-12)

        # --- simplified effective N (no roots / piecewise) ---
        K_I_safe = jnp.maximum(params.K_I, 1e-12)
        N_eff = N * (K_I_safe / (K_I_safe + L))  # receptor-competition-like throttle

        # activator production: Hill in N_eff
        N_act = params.sigma_N * hill01(N_eff / params.K_A, params.n)

        # repressor production: Hill in N_eff (keep as before, just swap argument)
        L_arg = jnp.where(params.apply_to_L, N_eff, N) / params.K_A
        L_prod = params.sigma_L * (
                jnp.power(jnp.clip(L_arg, 0, 1e6), params.p) /
                (jnp.power(params.K_NL / params.K_A, params.p) +
                 jnp.power(jnp.clip(L_arg, 0, 1e6), params.p))
        )

        N_loss = params.mu_N * N
        L_loss = params.mu_L * L

        # total sources
        S_N = S_N_dir + S_N_imp
        S_L = S_L_dir + S_L_imp

        dN += N_act - N_loss + S_N
        dL += L_prod - L_loss + S_L

        # sanitize (keeps numerical nasties from propagating)
        dN = jnp.nan_to_num(dN, nan=0.0, posinf=0.0, neginf=0.0)
        dL = jnp.nan_to_num(dL, nan=0.0, posinf=0.0, neginf=0.0)

        dy = jnp.concatenate([dN, dL, drho, dF_N, dF_L])
        return dy

    return rhs

# ----------------------------
# Initialization helpers
# ----------------------------
def init_gaussian(x: jnp.ndarray, amp: float, sigma: float, x0: float | None = None):
    if x0 is None:
        x0 = 0.5 * (x[0] + x[-1])
    return amp * jnp.exp(-0.5 * ((x - x0) / jnp.maximum(sigma, 1e-12)) ** 2)

def init_multi_gaussian(x: jnp.ndarray, amp: float, sigma: float, n_spots: int=1, key=None):
    if key is None:
        raise ValueError("Need a JAX PRNG key for random initialization")
    if n_spots == 1:
        centers = [0.5 * (x[0] + x[-1])]
    else:
        centers = jax.random.uniform(key, shape=(n_spots,), minval=x[0], maxval=x[-1])

    # Stack each Gaussian along a new axis, then sum over that axis
    gaussians = jnp.stack([
        amp * jnp.exp(-0.5 * ((x - c) / jnp.maximum(sigma, 1e-12)) ** 2)
        for c in centers
    ], axis=0)

    return jnp.sum(gaussians, axis=0)

def init_constant(x: jnp.ndarray, value: float):
    return jnp.full_like(x, value)

def build_initial_state_1d(x: jnp.ndarray, *,
                           N_mode="gaussian", N_amp=10.0, N_sigma=25.0, n_N_spots=1, N_key=None,
                           L_mode="constant", L_amp=0.0, n_L_spots=1, L_key=None,
                           rho_value=0.1):

    if N_key is None:
        N_key = jax.random.PRNGKey(np.random.randint(0, 1e9))
    if L_key is None:
        L_key = jax.random.PRNGKey(np.random.randint(0, 1e9))

    if (N_mode == "gaussian") and (n_N_spots == 1):
        N0 = init_gaussian(x, N_amp, N_sigma)
    elif (N_mode == "gaussian") and (n_N_spots > 1):
        N0 = init_multi_gaussian(x, N_amp, N_sigma, n_N_spots, N_key)
    elif N_mode == "constant":
        N0 = init_constant(x, N_amp)
    else:
        raise ValueError("Only 'gaussian' and 'constant' shown for brevity.")

    if L_mode == "constant":
        L0 = init_constant(x, L_amp)
    elif (L_mode == "gaussian") and (n_L_spots == 1):
        L0 = init_gaussian(x, L_amp, N_sigma)
    elif (L_mode == "gaussian") and (n_L_spots > 1):
        L0 = init_multi_gaussian(x, L_amp, N_sigma, n_L_spots, L_key)
    else:
        raise ValueError("Only 'constant' and 'gaussian' shown for brevity.")

    rho0 = init_constant(x, rho_value)

    # F_N, F_L start at 0
    F_N0 = jnp.zeros_like(x)
    F_L0 = jnp.zeros_like(x)

    y0 = jnp.concatenate([N0, L0, rho0, F_N0, F_L0])
    return y0

# ----------------------------
# Runner
# ----------------------------
def run_1d(params: Params1D,
           blips: BlipSet1D,
           T: float,
           nx: int,
           save_ts: jnp.ndarray,
           y0_override: jnp.ndarray | None = None):

    x = jnp.linspace(0.0, params.Lx, nx)
    # dx = jnp.asarray(params.dx, dtype=x.dtype)

    # ok = jnp.isclose(params.Lx / (nx - 1), dx)
    # jax.debug.print("WARNING: dx != Lx/(nx-1). dx={:.6g} Lx/(nx-1)={:.6g}", dx, params.Lx / (nx - 1), where=~ok)

    rhs = build_rhs_1d(x, params, blips)
    y0 = y0_override if y0_override is not None else build_initial_state_1d(
        x, N_mode="gaussian", N_amp=10.0, N_sigma=10.0, L_mode="constant", L_value=0.0, rho_value=0.1
    )

    term = dfx.ODETerm(rhs)
    solver = dfx.Dopri5()
    controller = dfx.PIDController(rtol=1e-5, atol=1e-8)# stiff, adaptive
    sol = dfx.diffeqsolve(
        term, solver,
        t0=0.0, t1=T, dt0=1e-3,
        y0=y0, args=None,
        saveat=dfx.SaveAt(ts=save_ts),
        stepsize_controller=controller,
        max_steps=10_000_000
    )
    return x, sol.ts, sol.ys  # ys has shape (len(save_ts), 5*nx)
