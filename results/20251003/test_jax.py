import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from symmetry_breaking_JAX.models_2D.run import simulate_2d, build_initial_state_2d, run_2d
from symmetry_breaking_JAX.models_2D.geom_utils import make_grid
from symmetry_breaking_JAX.models_2D.param_classes import Params2D, BlipSet2D

T = 1000

# 1. Define parameters and blips
params = Params2D(
    D_N=1.0, D_L=15.0,
    sigma_N=2.0, sigma_L=1.0,
    mu_N=1.0, mu_L=1.0,
    n=2, p=2, alpha=1.0,
    K_A=667.0, K_NL=667.0, K_I=1.0,
    Lx=1500.0, Ly=1500.0, dx=4.0, dy=4.0,
    geometry="disk",
)

blips = BlipSet2D.empty()  # no dynamic events for this test

# 2. Build grid
grid = make_grid(params,)

# 3. Initialize state
y0 = build_initial_state_2d(
    params, grid,
    N_mode="gaussian",
    N_positions=jnp.array([[0, 0], [50, 50]]),
    N_amps=jnp.array([1000, 500]),
    N_sigmas=jnp.array([25, 25]),
    L_mode="constant",
    L_amp=0.0,
)

# 4. Integrate
ts, ys = run_2d(
    params,
    blips,
    grid,
    T=T,
    save_ts=jnp.linspace(0, T, 200),
    y0=y0,
)

# unpack grid
X, Y, mask = grid.X, grid.Y, grid.mask
nx, ny = X.shape
nxy = nx * ny

# extract Nodal field over time
N_t = ys[:, :nxy].reshape(len(ts), nx, ny)

# ----------------------------
# 4. Visualization (real-time heatmap)
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(N_t[-1], cmap="viridis", origin="lower",
               extent=[X.min(), X.max(), Y.min(), Y.max()],
               vmin=0, vmax=N_t.max())
ax.set_title(f"t = {ts[0]:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame):
    im.set_data(N_t[frame])
    ax.set_title(f"t = {ts[frame]:.2f}")
    return [im]

ani = FuncAnimation(fig, update, frames=len(ts), interval=100, blit=True)
plt.show()

print("check")
