import napari
import os
import pyefd
import pandas as pd
from tqdm import tqdm
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
from skimage.measure import regionprops
import zarr
from skimage.filters import gaussian
import skimage.io as io
from pyefd import elliptic_fourier_descriptors
import skimage
from skimage.morphology import label
import json
from skimage.filters import threshold_multiotsu
from skimage.transform import resize
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

# # set parameter


def refine_cell_masks(root, project_name, config_name, overwrite_flag=False):

    tracking_folder = config_name.replace(".txt", "")
    tracking_folder = tracking_folder.replace(".toml", "")

    tracking_directory = os.path.join(root, "built_data", "tracking", project_name, tracking_folder)
    snip_directory = os.path.join(root, "built_data", "shape_snips", project_name, tracking_folder, "class0", "")
    snip_directory_refined = os.path.join(root, "built_data", "shape_snips_refined", project_name, tracking_folder, "class0", "")
    if not os.path.isdir(snip_directory_refined):
        os.makedirs(snip_directory_refined)
    im_directory = os.path.join(root, "built_data", "shape_images", project_name, tracking_folder, "class0", "")

    # load track and segment info
    cfg = load_config(os.path.join(root, "metadata", project_name, config_name))
    tracks_df, graph = to_tracks_layer(cfg)
    # load sphere fit info
    # sphere_df = pd.read_csv(os.path.join(root, "metadata", project_name, "sphere_df.csv"))

    print("loading cell shape masks...")
    track_index = np.unique(tracks_df["track_id"])
    # coeff_cols = []
    # for n in range(n_shape_coeffs):
    #     for c in range(4):
    #         coeff_string = "coeff_" + f'row{n:02}_' + f'col{c:01}'
    #         coeff_cols.append(coeff_string)

    # coeff_cols = coeff_cols[3:]
    df_list = []
    for _, track_id in enumerate([5]):#tqdm(track_index)):

        # iterate through label masks
        cell_df = tracks_df.loc[tracks_df["track_id"] == track_id, ["track_id", "t"]].copy()

        for i, ind in enumerate(cell_df.index):
            # extrack mask info from tracks
            time_id = cell_df.loc[ind, "t"]
            # centroid = frame_df.loc[ind, ["z", "y", "x"]].to_numpy()

            # make read/write name
            snip_name = f'snip_track{track_id:04}_t{time_id:04}.jpg'
            snip_path = os.path.join(snip_directory, snip_name)
            snip_path_out = os.path.join(snip_directory_refined, snip_name)

            if (not os.path.exists(snip_path_out)) | overwrite_flag:
                im_name = f'im_track{track_id:04}_t{time_id:04}.jpg'
                im_path = os.path.join(im_directory, im_name)

                # load cell snip
                snip = io.imread(snip_path)
                im = io.imread(im_path)

                # get shape descriptor
                snip_bin = snip > 50

                mask_rs = resize(snip_bin[8:-8, 8:-8], im.shape, preserve_range=True, order=0)

                if len(np.unique(im[mask_rs > 0])) > 3:
                    thresholds = threshold_multiotsu(im[mask_rs > 0], 3)
                    thresh = thresholds[0]
                    n_regions = np.inf
                    while n_regions > 1:
                        im_thresh = im > thresh
                        im_lb = label(im_thresh)
                        n_regions = np.max(im_lb)
                        thresh -= 1

                        if thresh == 0:
                            break

                    mask_new = np.zeros(im.shape, dtype=np.uint8)
                    temp_mask = im > (thresh + 1)
                    temp_mask[mask_rs == 0] = 0
                    lb_mask = label(temp_mask)
                    if np.max(lb_mask) > 1:
                        rg = regionprops(lb_mask)
                        area_vec = np.asarray([r["area"] for r in rg])
                        keep_i = rg[np.argmax(area_vec)].label
                    else:
                        keep_i = 1
                    mask_new[lb_mask == keep_i] = 255
                    mask_new = skimage.morphology.area_closing(mask_new)
                else:
                    mask_new = mask_rs

                if time_id == 273:
                    print("check")

                # if time_id == 38:
                #     print("check")
                io.imsave(snip_path_out, mask_new, check_contrast=False)

if __name__ == '__main__':

    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "240219_LCP1_93hpf_to_127hpf"  # "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw" #
    overwrite_flag = True
    config_name = "tracking_cell.txt"
    refine_cell_masks(root, project_name, config_name, overwrite_flag=overwrite_flag)