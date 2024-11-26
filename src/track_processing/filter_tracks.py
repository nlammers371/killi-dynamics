import napari
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
import zarr
import dask.array as da
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

# # set parameters
# root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
experiment_date = "20240611_NLS-Kikume_24hpf_side2" #"230425_EXP21_LCP1_D6_1pm_DextranStabWound" #
config_name = "tracking_jordao_20240918.txt" #"tracking_jordao_full" #
model ="LCP-Multiset-v1"
tracking_folder = config_name.replace(".txt", "")
tracking_folder = tracking_folder.replace(".toml", "")

killi_flag = True
well_num = 0
start_i = 0
stop_i = 1600

suffix = ""

# get path to metadata
metadata_path = os.path.join(root, "metadata", "tracking")

# set output path for tracking results
project_path = os.path.join(root, "tracking", experiment_date,  tracking_folder, f"well{well_num:04}" + suffix, "")
project_sub_path = os.path.join(project_path, f"track_{start_i:04}" + f"_{stop_i:04}" + suffix, "")

# path to image data
data_path = os.path.join(root, "built_data", "cellpose_output", model, experiment_date, "")
filename = experiment_date + "_probs.zarr"


if experiment_date == "230425_EXP21_LCP1_D6_1pm_DextranStabWound":
    project_sub_path = project_path

################
# load tracking results
prob_path = os.path.join(data_path, filename)
label_path = os.path.join(project_sub_path, "segments.zarr")
prob_zarr = zarr.open(prob_path, mode='r')
seg_zarr = zarr.open(label_path, mode='r')
tracks_df = pd.read_csv(os.path.join(project_sub_path, "tracks.csv"))

# get scale
scale_vec = np.asarray(prob_zarr.attrs["voxel_size_um"])
voxel_vol = np.prod(scale_vec)

# count pixels (volume) in each image
df_list = []
for t in tqdm(range(start_i, stop_i)):
    seg_flat = seg_zarr[t].ravel()
    lc_full = np.bincount(seg_flat[seg_flat > 0])
    lu = np.where(lc_full)[0]
    lc = lc_full[lu]

    df_temp = pd.DataFrame([t] * len(lu), columns=["t"])
    df_temp["track_id"] = lu
    df_temp["volume"] = lc * voxel_vol
    df_list.append(df_temp)

# combine
size_df = pd.concat(df_list, axis=0, ignore_index=True)

# merge
tracks_df = tracks_df.merge(size_df, how="left", on=["track_id", "t"], indicator=False)

# calculate velocity
tracks_df_v = tracks_df.copy()
tracks_df_v[["dx", "dy", "dz", "dV"]] = tracks_df_v.loc[:, ["track_id", "parent_track_id", "x", "y", "z", "volume"]].groupby(["track_id", "parent_track_id"]).diff()
tracks_df_v["v"] = np.sqrt(tracks_df_v["dx"]**2 + tracks_df_v["dy"]**2 + tracks_df_v["dz"]**2)

# calculate average velocity and volume for each cell. We will use this for QC
tracks_df_v = tracks_df_v.drop(labels=["dV", "id", "parent_id"], axis=1).dropna()
tracks_df_v = tracks_df_v.groupby(["track_id", "parent_track_id"]).mean().reset_index()

# fit a 2-component GMM to segment EVL from deep cells
vv_array = tracks_df_v[["volume", "v"]].to_numpy()
vv_array = vv_array - np.mean(vv_array, axis=0)
vv_array = np.divide(vv_array, np.percentile(np.abs(vv_array), 95, axis=0))

# Fit a Gaussian Mixture Model with 3 components
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(vv_array)

# Extract results
labels = gmm.predict(vv_array)  # Cluster labels for each point
probs = gmm.predict_proba(vv_array)
scores = gmm.score_samples(vv_array)

# update tracks
tracks_df_v["gmm_label"] = labels
tracks_df_v[["gmm0_prob", "gmm1_prob"]] = probs
tracks_df_v["gmm_logL"] = scores

# merge onto the full data frame
tracks_df = tracks_df.merge(tracks_df_v.loc[:, ["track_id", "gmm_label", "gmm0_prob", "gmm1_prob", "gmm_logL"]],
                            how="left", on="track_id")

tracks_df.to_csv(os.path.join(project_sub_path, "tracks01.csv"), index=False)


time_index = np.unique(tracks_df["time"])


t_min = 1100
t_max = 1110
time_filter = (tracks_df["t"] >= t_min) & (tracks_df["t"] < t_max)
g0_filter = tracks_df["gmm0_prob"] >= 0.5
g1_filter = (tracks_df["gmm0_prob"] < 0.5) & (tracks_df["gmm0_prob"] >= 0.3)
g2_filter = tracks_df["gmm0_prob"] < 0.3

g_ids0 = tracks_df.loc[g0_filter, "track_id"].to_numpy()
g_ids1 = tracks_df.loc[g1_filter, "track_id"].to_numpy()
g_ids2 = tracks_df.loc[g2_filter, "track_id"].to_numpy()

seg_short = seg_zarr[t_min:t_max]
seg_lb = np.zeros_like(seg_short)
seg_score = np.zeros(seg_lb.shape, dtype=np.float16)

t_list = np.arange(t_min, t_max)
for t, time in enumerate(tqdm(t_list)):

    seg_lb[t][np.isin(seg_short[t], g_ids0)] = 1
    seg_lb[t][np.isin(seg_short[t], g_ids1)] = 2
    seg_lb[t][np.isin(seg_short[t], g_ids2)] = 3

#
viewer = napari.Viewer(ndisplay=3) #view_image(data_zarr_da, scale=tuple(scale_vec))
# viewer.add_tracks(
#     tracks_df[["track_id", "t", "z", "y", "x"]],
#     name="tracks",
#     scale=tuple(scale_vec),
#     translate=(0, 0, 0, 0),
#     visible=False,
# )
# viewer.scale_bar.visible = True
# viewer.scale_bar.unit = "um"

viewer.add_image(
    prob_zarr[t_min:t_max],
    name="probs",
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
).contour = 2

viewer.add_labels(
    seg_short,
    name="segments",
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
).contour = 2

viewer.add_labels(
    seg_lb,
    name="segments-thresh",
    scale=tuple(scale_vec),
    translate=(0, 0, 0, 0),
).contour = 2
#
# # filter for the best tracks
# track_index, track_counts = np.unique(tracks_df["track_id"], return_counts=True)
# min_len = 40
# good_tracks = track_index[track_counts >= min_len]
#
#
# tracks_df_qc = tracks_df.loc[np.isin(tracks_df["track_id"], good_tracks), :]
# seg_qc = np.asarray(seg_zarr).copy()
# seg_qc[~np.isin(seg_qc, good_tracks)] = 0
#
#
# viewer.add_tracks(
#     tracks_df_qc[["track_id", "t", "z", "y", "x"]],
#     name="tracks qc",
#     scale=tuple(scale_vec),
#     translate=(0, 0, 0, 0),
#     visible=False,
# )
#
#
# viewer.add_labels(
#     seg_qc,
#     name="segments qc",
#     scale=tuple(scale_vec),
#     translate=(0, 0, 0, 0),
# ).contour = 2


# viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')

if __name__ == '__main__':
    napari.run()