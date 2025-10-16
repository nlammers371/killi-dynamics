import napari
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
# from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
import zarr
import dask.array as da
from tqdm import tqdm

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
raw_data_path = os.path.join(root, "built_data", "zarr_image_files", experiment_date, "")

if not killi_flag:
    filename = experiment_date + f"_well{well_num:04}_probs.zarr"
else:
    filename = experiment_date + "_probs.zarr"
# data_path = os.path.join(root, "built_data", "zarr_image_files", experiment_date, "")
# filename = experiment_date + ".zarr"

if experiment_date == "230425_EXP21_LCP1_D6_1pm_DextranStabWound":
    project_sub_path = project_path

# load tracking results
# image_path = os.path.join(raw_data_path[:-1] + ".zarr")
image_path = os.path.join(data_path, filename)
label_path = os.path.join(project_sub_path, "segments.zarr")


data_zarr = zarr.open(image_path, mode='r')

scale_vec = tuple([1.5, 1.5, 1.5]) #data_zarr.attrs["voxel_size_um"])

seg_zarr = zarr.open(label_path, mode='r')

tracks_df = pd.read_csv(os.path.join(project_sub_path, "tracks.csv"))

# subsample to test some ideas...

df_list = []
for t in tqdm(range(start_i, stop_i)):
    nz_ids = seg_zarr[t] > 0
    id_vec = seg_zarr[t][nz_ids]
    f_vec = data_zarr[t][nz_ids]
    df_temp = pd.DataFrame([t]*len(id_vec), columns=["t"])
    df_temp["track_id"] = id_vec
    df_temp["fluo"] = f_vec
    df_list.append(df_temp)

nucleus_df = pd.concat(df_list, axis=0, ignore_index=True)

size_df = nucleus_df.groupby(["time", "nucleus_id"]).size().rename("Volume").reset_index()

tracks_df = tracks_df.merge(size_df, how="left", on=["track_id", "t"], indicator=False)
# tracks_df.to_csv(os.path.join(project_sub_path, "tracks_size.csv"), index=False)
#
# time_index = np.unique(size_df["time"])
#
# data = size_df_short["Volume"]
# thresh = 1050

# nbins = 10
# group_ids = (pd.qcut(data, q=nbins, labels=False) + 1).to_numpy()
# group_index = np.unique(group_ids)
# size_df["total_group"] = group_ids

# for id in group_index:

# g_filter1 = data <= thresh - 50
# g_filter2 = (data <= thresh) & (data > thresh - 50)
# g_filter3 = (data <= thresh + 50) & (data > thresh)
# g_filter4 = data > thresh + 50
#
# g_ids1 = size_df_short.loc[g_filter1, "nucleus_id"].to_numpy()
# g_ids2 = size_df_short.loc[g_filter2, "nucleus_id"].to_numpy()
# g_ids3 = size_df_short.loc[g_filter3, "nucleus_id"].to_numpy()
# g_ids4 = size_df_short.loc[g_filter4, "nucleus_id"].to_numpy()
#
# t_list = np.arange(t_min, t_max)
# for t, time in enumerate(tqdm(t_list)):
#
#     seg_lb[t][np.isin(seg_zarr[time], g_ids1)] = 1
#     seg_lb[t][np.isin(seg_zarr[time], g_ids2)] = 2
#     seg_lb[t][np.isin(seg_zarr[time], g_ids3)] = 3
#     seg_lb[t][np.isin(seg_zarr[time], g_ids4)] = 4
#
viewer = napari.Viewer(ndisplay=3) #view_image(data_zarr_da, scale=tuple(scale_vec))

viewer.add_image(
    image_data[0],
    name="probs",
    scale=tuple(scale_vec),
    # translate=(0, 0, 0, 0),
).contour = 2

# viewer.add_labels(
#     seg_lb,
#     name="segments-thresh",
#     scale=tuple(scale_vec),
#     translate=(0, 0, 0, 0),
# ).contour = 2
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