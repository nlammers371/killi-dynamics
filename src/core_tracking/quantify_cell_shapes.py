import napari
import os
import pyefd
import pandas as pd
from tqdm import tqdm
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
import zarr
from skimage.filters import gaussian
import skimage.io as io
from pyefd import elliptic_fourier_descriptors
import skimage
import json
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

# # set parameters
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project_name = "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
image_dir = os.path.join(root, "built_data", project_name, "")
snip_dim = 64
overwrite_flag = False
config_name = "tracking_final.txt"
tracking_folder = config_name.replace(".txt", "")
tracking_folder = tracking_folder.replace(".toml", "")
n_shape_coeffs = 10

tracking_directory = os.path.join(root, "built_data", "tracking", tracking_folder, project_name)
snip_directory = os.path.join(root, "built_data", "shape_snips", tracking_folder, project_name)

# load metadata
metadata_file_path = os.path.join(root, "metadata", project_name, "metadata.json")
f = open(metadata_file_path)
metadata = json.load(f)
scale_vec = np.asarray([metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])

# make snip ref grids
y_ref_snip, x_ref_snip = np.meshgrid(range(snip_dim),
                                      range(snip_dim),
                                      indexing="ij")

# load track and segment info
cfg = load_config(os.path.join(root, "metadata", project_name, config_name))
tracks_df, graph = to_tracks_layer(cfg)
segments = zarr.open(os.path.join(tracking_directory, "segments.zarr"), mode='r')

# load sphere fit info
sphere_df = pd.read_csv(os.path.join(root, "metadata", project_name, "sphere_df.csv"))

print("loading cell shape masks...")
track_index = np.unique(tracks_df["track_id"])
coeff_cols = []
for n in range(n_shape_coeffs):
    for c in range(4):
        coeff_string = "coeff_" + f'row{n:02}_' + f'col{c:01}'
        coeff_cols.append(coeff_string)
        
coeff_cols = coeff_cols[3:]
df_list = []
for t, track_id in enumerate(tqdm(track_index)):

    # iterate through label masks
    cell_df = tracks_df.loc[tracks_df["track_id"] == track_id, ["track_id", "t"]].copy()
    
    for i, ind in enumerate(cell_df.index):
        # extrack mask info from tracks
        time_id = cell_df.loc[ind, "t"]
        # centroid = frame_df.loc[ind, ["z", "y", "x"]].to_numpy()

        # make read/write name
        snip_name = f'snip_track{track_id:04}_t{time_id:04}.jpg'
        snip_path = os.path.join(snip_directory, snip_name)

        # load cell snip
        snip = io.imread(snip_path)

        # get shape descriptor
        snip_bin = snip > 50
        contour = skimage.measure.find_contours(snip_bin, 0.5)

        # get shape coefficients
        coeffs = elliptic_fourier_descriptors(
            contour[0], order=10, normalize=True)

        out = coeffs.flatten()[3:]
        cell_df.loc[ind, coeff_cols] = out

    df_list.append(cell_df)
        
shape_df = pd.concat(df_list, axis=0, ignore_index=True)
shape_df.to_csv(os.path.join(tracking_directory, "cell_shape_df.csv"), index=False)
#if__name__ == '__main__':
