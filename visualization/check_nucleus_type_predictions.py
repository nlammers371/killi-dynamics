import zarr
import napari
import numpy as np
from src.build_killi.build_utils import fit_sphere_and_sh, create_sphere_mesh, create_sh_mesh
import pandas as pd
import statsmodels.api as sm
import os
from tqdm import tqdm
from pathlib import Path

os.environ["QT_API"] = "pyqt5"

# get filepaths
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project_name = "20250419_BC1-NLSMSC"
tracking_config = "tracking_20250328_redux"

start_i = 0
stop_i = 614
tracking_range = [start_i, stop_i]
suffix = ""
scale_vec = np.asarray([3.0, 0.85, 0.85])
well_num = 0
par_flag = True

print("Loading tracking data for project:", project_name)

# load mask features
project_path = os.path.join(root, "tracking", project_name, tracking_config, f"well{well_num:04}", "")
save_path = os.path.join(project_path, f"track_{tracking_range[0]:04}" + f"_{tracking_range[1]:04}" + suffix, "")

# read mask data
mask_path = Path(save_path) / "segments.zarr"
mask = zarr.open(mask_path, mode="r")

# read in classification data
class_df = pd.read_csv(Path(save_path) / "track_class_df_full.csv")

t_range = [100, 110]

# subset class_df
class_df = class_df.loc[(class_df["t"] >= t_range[0]) & (class_df["t"] < t_range[1])]
# generate array to transfer labels
mask_to_plot = mask[t_range[0]:t_range[1]]
labels = np.zeros(mask_to_plot.shape, dtype=np.uint8)
labels_u = class_df["track_class"].unique()

for t, time in enumerate(tqdm(range(t_range[0], t_range[1]))):
    lb_df = class_df.loc[class_df["t"] == time, ["track_id", "track_class"]]
    mask_slice = mask_to_plot[t]
    lb_slice = labels[t]
    for label in labels_u:
        # get indices of current label
        idx = lb_df.loc[lb_df["track_class"] == label, "track_id"].to_numpy()
        lb_slice[np.isin(mask_slice, idx)] = label + 1

    labels[t] = lb_slice



viewer = napari.Viewer()

viewer.add_labels(mask_to_plot, scale=scale_vec)
viewer.add_labels(labels, scale=scale_vec)

napari.run()

print("Check")

