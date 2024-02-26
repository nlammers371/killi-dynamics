import napari
import os

import pandas as pd
from tqdm import tqdm
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
import zarr
from skimage.filters import gaussian
import skimage.io as io
from skimage.transform import resize
from src.utilities.functions import sphereFit
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import binary_closing, disk
import json
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

# # set parameters


def extract_cell_masks(root, project_name, config_name, overwrite_flag=False, snip_dim=64):

    tracking_folder = config_name.replace(".txt", "")
    tracking_folder = tracking_folder.replace(".toml", "")
    # image_dir = os.path.join(root, "built_data", project_name, "")

    print("loading track and mask info...")
    read_directory = os.path.join(root, "built_data", "tracking", project_name, tracking_folder)
    save_directory = os.path.join(root, "built_data", "shape_snips",  project_name, tracking_folder)
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    metadata_file_path = os.path.join(root, "metadata", project_name, "metadata.json")
    f = open(metadata_file_path)
    metadata = json.load(f)
    scale_vec = np.asarray([metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])

    # make snip ref grids
    # y_ref_snip, x_ref_snip = np.meshgrid(range(snip_dim),
    #                                       range(snip_dim),
    #                                       indexing="ij")

    morph_fp = disk(2)
    # load track and segment info
    cfg = load_config(os.path.join(root, "metadata", project_name, config_name))
    tracks_df, graph = to_tracks_layer(cfg)
    segments = zarr.open(os.path.join(read_directory, "segments.zarr"), mode='r')

    # path to raw cell labels
    raw_mask_zarr = os.path.join(root, "built_data", "cleaned_cell_labels", project_name + ".zarr")
    raw_masks = zarr.open(raw_mask_zarr, mode='r')
    # mask_list = sorted(glob.glob(raw_mask_directory + "*.tif"))

    # load sphere fit info
    sphere_df = pd.read_csv(os.path.join(root, "metadata", project_name, "sphere_df.csv"))

    print("generating 2D cell profiles...")

    for t in tqdm(range(segments.shape[0])):
        # raw_mask_arr = io.imread(mask_list[t])
        raw_mask_arr = np.asarray(raw_masks[t])
        # iterate through label masks
        frame_df = tracks_df.loc[tracks_df["t"] == t, :]
        frame_arr = np.asarray(segments[t])
        sphere_center_um = sphere_df.loc[sphere_df["t"] == t, ["Z", "Y", "X"]].to_numpy()[0]
        sphere_center = np.divide(sphere_center_um, scale_vec)

        if t == 0:
            z_ref, y_ref, x_ref = np.meshgrid(range(frame_arr.shape[0]),
                                              range(frame_arr.shape[1]),
                                              range(frame_arr.shape[2]),
                                              indexing="ij")

        foreground_mask = (frame_arr > 0) | (raw_mask_arr > 0)
        ultrack_lb_vec = frame_arr[foreground_mask]
        raw_lb_vec = raw_mask_arr[foreground_mask]
        zyx_arr = np.concatenate((np.reshape(z_ref[foreground_mask], (len(ultrack_lb_vec), 1)),
                                 np.reshape(y_ref[foreground_mask], (len(ultrack_lb_vec), 1)),
                                 np.reshape(x_ref[foreground_mask], (len(ultrack_lb_vec), 1))), axis=1)


        for i, ind in enumerate(frame_df.index):
            # extrack mask info from tracks
            track_id = frame_df.loc[ind, "track_id"]
            centroid = frame_df.loc[ind, ["z", "y", "x"]].to_numpy()

            # make read/write name
            snip_name = f'snip_track{track_id:04}_t{t:04}.jpg'
            snip_path = os.path.join(save_directory, snip_name)

            if (not os.path.isfile(snip_path)) | overwrite_flag:

                # get normal vector
                normal_vec = centroid - sphere_center
                norm_u = normal_vec / np.sqrt(np.sum(normal_vec**2))

                # get coordinates for points within the cell
                raw_lbs = np.unique(raw_lb_vec[ultrack_lb_vec == track_id])
                lb_mask = np.asarray([raw_lb_vec[i] in raw_lbs for i in range(len(raw_lb_vec))])
                zyxp = zyx_arr[lb_mask, :]

                # find d for perp plane
                d = np.sum(np.multiply(norm_u, centroid))

                # get scalar distances to plane
                # ds = np.sum(np.multiply(norm_u[np.newaxis, :], zyxp), axis=1) - d

                # center
                zyxp_plane = zyxp - centroid #np.multiply(ds[:, np.newaxis], np.tile(norm_u[np.newaxis, :], (len(ds), 1)))

                # convert centroid to spherical
                centroid_sph = cartesian_to_spherical(centroid[2], centroid[1], centroid[0])
                new_point_sph = np.asarray([c.value for c in centroid_sph])
                new_point_sph[1] = new_point_sph[1] + 0.1 # increment in phi direction

                # convert back to cartesian
                new_point_cart = spherical_to_cartesian(new_point_sph[0], new_point_sph[1], new_point_sph[2])
                new_point = np.asarray([c.value for c in new_point_cart])
                new_point = new_point[::-1]
                plane_point = new_point.copy()
                plane_point[0] = (-norm_u[1]*plane_point[1] - norm_u[2]*plane_point[2] + d) / norm_u[0]

                # calculate unit vectors to define my two snip axes
                v1 = centroid - plane_point
                v1 = v1 / np.sqrt(np.sum(v1**2))
                v2 = np.cross(v1, norm_u)
                v2 = v2 / np.sqrt(np.sum(v2 ** 2))

                # take dot product to convert to plane coordinates
                c1 = np.round(np.sum(np.multiply(v1[np.newaxis], zyxp_plane), axis=1)).astype(np.uint8) + snip_dim // 2 + 1
                c2 = np.round(np.sum(np.multiply(v2[np.newaxis], zyxp_plane), axis=1)).astype(np.uint8) + snip_dim // 2 + 1

                # convert to snip
                snip_raw = np.zeros((snip_dim, snip_dim)).astype(np.uint8)
                snip_raw[c1, c2] = 1
                # snip_dist = distance_transform_edt(snip_raw != 1)
                # snip_dist[snip_dist > 1] = 1

                snip = (binary_closing(snip_raw > 0, morph_fp) * 255).astype(np.uint8)

                io.imsave(snip_path, snip)




if __name__ == '__main__':
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"

    snip_dim = 64
    overwrite_flag = True
    config_name = "tracking_final.txt"

    extract_cell_masks(root, project_name, config_name, overwrite_flag=True)
