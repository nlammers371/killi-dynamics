import numpy as np, json
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")   # Non-interactive backend (saves figures to file)
from matplotlib import pyplot as plt

root = Path("/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/symmetry_breaking/pde/sweeps/")
output_dir = root / "sweep01_jax_stable"


# Reload arrays
data = np.load(output_dir / "results.npz")
N = data["N"]
L = data["L"]

# Reload parameters
with open(output_dir / "params.json") as f:
    param_dicts = json.load(f)

print(N.shape, L.shape, len(param_dicts))

sim_idx = 9947   # <-- change this
data = np.array(N[sim_idx])  # shape (time, space)

fig, ax = plt.subplots(figsize=(8, 6))

# imshow expects (rows, cols) = (time, space)
im = ax.imshow(
    data,
    aspect="auto",              # stretch to fill axes
    origin="lower",             # time goes upward
    cmap="viridis",             # or 'plasma', 'inferno', etc.
    extent=[0, data.shape[1],   # x-axis: gridpoints
            0, data.shape[0]]   # y-axis: timepoints
)

ax.set_xlabel("Space (grid index)")
ax.set_ylabel("Time (frame index)")
ax.set_title(f"Nodal kymograph, simulation {sim_idx}")

fig.colorbar(im, ax=ax, label="Concentration")
plt.show()

print("Check")