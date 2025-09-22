# test_blips.py
import jax.numpy as jnp
import numpy as np
from symmetry_breaking_JAX.models.JAX_NL_1D import Params1D, BlipSet1D, run_1d, build_initial_state_1d

# -----------------------------
# Parameters (dimension-full)
# -----------------------------
params = Params1D(
    D_N=1.85, D_L=15.0,
    sigma_N=10.0, sigma_L=0.1,
    mu_N=1e-4, mu_L=1e-4,
    n=2, p=2, alpha=1.0,
    K_A=667.0, K_NL=667.0, K_I=1.0,
    Lx=1500.0, dx=10.0, bc="neumann",
    tau_impulse=60.0,
    sigma_t_direct=1e-3  # << super small for near-instantaneous blips
)

# -----------------------------
# Pre-sample blips
# -----------------------------
# Example: two direct Nodal blips
N_dir_times = jnp.array([])   # seconds
N_dir_x     = jnp.array([])    # microns
N_dir_amp   = jnp.array([])      # amplitude (area in time)
N_dir_sigx  = jnp.array([])      # spatial sigma (µm)

# No impulse-mode Nodal blips
N_imp_times = jnp.array([])
N_imp_x     = jnp.array([])
N_imp_amp   = jnp.array([])
N_imp_sigx  = jnp.array([])

# Constant Lefty background (no blips)
L_dir_times = jnp.array([])
L_dir_x     = jnp.array([])
L_dir_amp   = jnp.array([])
L_dir_sigx  = jnp.array([])

L_imp_times = jnp.array([])
L_imp_x     = jnp.array([])
L_imp_amp   = jnp.array([])
L_imp_sigx  = jnp.array([])

blips = BlipSet1D(
    N_dir_times, N_dir_x, N_dir_amp, N_dir_sigx,
    N_imp_times, N_imp_x, N_imp_amp, N_imp_sigx,
    L_dir_times, L_dir_x, L_dir_amp, L_dir_sigx,
    L_imp_times, L_imp_x, L_imp_amp, L_imp_sigx
)

# -----------------------------
# Initial conditionfrhos
# -----------------------------
nx = params.Lx / params.dx + 1
x = jnp.linspace(0.0, params.Lx, nx)

# Start with Gaussian Nodal pulse, constant Lefty background
y0 = build_initial_state_1d(
    x,
    N_mode="gaussian", N_amp=20.0, N_sigma=50.0,
    L_mode="constant", L_amp=0.0,    # << constant Lefty background
    rho_value=0.1
)

# -----------------------------
# Time span & outputs
# -----------------------------
T = 10 * 3600.0  # 10 h
# Save every 5 min plus EXACT blip times
save_regular = jnp.linspace(0.0, T, 121)     # ~5 min spacing
save_ts = jnp.sort(jnp.concatenate([save_regular, N_dir_times]))

# -----------------------------
# Run simulation
# -----------------------------
x, ts, ys = run_1d(params, blips, T=T, nx=nx, save_ts=save_ts, y0_override=y0)

# -----------------------------
# Unpack results
# -----------------------------
N_t   = np.array(ys[:, 0:nx])
L_t   = np.array(ys[:, nx:2*nx])
rho_t = np.array(ys[:, 2*nx:3*nx])  # passive, unused

print("Simulation finished. Shapes:")
print("  times:", ts.shape)
print("  Nodal:", N_t.shape)
print("  Lefty:", L_t.shape)
