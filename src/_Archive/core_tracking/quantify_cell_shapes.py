import napari
import os
import pyefd
import pandas as pd
from tqdm import tqdm
import numpy as np
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import glob2 as glob
from skan.csr import skeleton_to_csgraph
from skimage.measure import regionprops
import zarr
from skimage.filters import gaussian
import skimage.io as io
from pyefd import elliptic_fourier_descriptors
import skimage
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from skimage.morphology import label
import json
from skimage.filters import threshold_multiotsu
from skimage.transform import resize
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

# # set parameters


def extract_cell_shape_stats(root, project_name, config_name, motion_orient_flag=True, shape_orient_flag=False):

    tracking_folder = config_name.replace(".txt", "")
    tracking_folder = tracking_folder.replace(".toml", "")
    n_shape_coeffs = 10

    tracking_directory = os.path.join(root, "built_data", "tracking", project_name, tracking_folder)

    if motion_orient_flag:
        suffix = "_motion_oriented"
    elif shape_orient_flag:
        suffix = "_shape_oriented"
    else:
        suffix = ""

    snip_directory = os.path.join(root, "built_data", "shape_snips_refined" + suffix, project_name, tracking_folder, "class0", "")
    im_directory = os.path.join(root, "built_data", "image_snips_refined" + suffix, project_name, tracking_folder, "class0", "")

    # load metadata
    # metadata_file_path = os.path.join(root, "metadata", project_name, "metadata.json")
    # f = open(metadata_file_path)
    # metadata = json.load(f)
    # scale_vec = np.asarray([metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])
    #
    # # make snip ref grids
    # y_ref_snip, x_ref_snip = np.meshgrid(range(snip_dim),
    #                                       range(snip_dim),
    #                                       indexing="ij")

    # load track and segment info

    # segments = zarr.open(os.path.join(tracking_directory, "segments.zarr"), mode='r')
    tracks_df = pd.read_csv(os.path.join(tracking_directory, "tracks_cleaned.csv"))

    # load sphere fit info
    # sphere_df = pd.read_csv(os.path.join(root, "metadata", project_name, "sphere_df.csv"))

    print("loading cell shape masks...")
    track_index = np.unique(tracks_df["track_id"])
    coeff_cols = []
    for n in range(n_shape_coeffs):
        for c in range(4):
            coeff_string = "coeff_" + f'row{n:02}_' + f'col{c:01}'
            coeff_cols.append(coeff_string)

    # coeff_cols = coeff_cols[3:]
    df_list = []
    for _, track_id in enumerate(tqdm(track_index)):

        # iterate through label masks
        cell_df = tracks_df.loc[tracks_df["track_id"] == track_id, ["track_id", "track_id_orig", "t"]].copy()

        for i, ind in enumerate(cell_df.index):
            # extrack mask info from tracks
            time_id = cell_df.loc[ind, "t"]
            track_id_orig = cell_df.loc[ind, "track_id_orig"]
            # centroid = frame_df.loc[ind, ["z", "y", "x"]].to_numpy()

            # make read/write name
            snip_name = f'snip_track{track_id_orig:04}_t{time_id:04}.jpg'
            snip_path = os.path.join(snip_directory, snip_name)

            im_name = f'im_track{track_id_orig:04}_t{time_id:04}.jpg'
            im_path = os.path.join(im_directory, im_name)

            # load cell snip
            snip = io.imread(snip_path)
            im = io.imread(im_path)

            # get shape descriptor
            snip_bin = snip > 50

            if np.sum(snip_bin) > 25:
                contour = skimage.measure.find_contours(snip_bin, 0.5)

                # get shape coefficients
                coeffs = elliptic_fourier_descriptors(contour[0], order=10, normalize=False)
                out = coeffs.flatten()
                cell_df.loc[ind, coeff_cols] = out

                # calculate more traditional metrics
                # perimeter = skimage.measure.perimeter_crofton(snip_bin)
                skeleton = skeletonize(snip_bin)
                if np.sum(skeleton) > 1:
                    # graph, pixel_coords = skeleton_to_csgraph(skeleton)
                    branch_table = summarize(Skeleton(skeleton))
                    n_branches = branch_table.shape[0]
                else:
                    n_branches = 0
                # branch_dist = np.mean(branch_table["branch-distance"])
                rg = regionprops(snip_bin.astype(int))
                cell_df.loc[ind, "n_branches"] = n_branches
                # cell_df.cloc[ind, "n_branches"] = branch_table.shape[0]
                cell_df.loc[ind, "perimeter"] = rg[0].perimeter
                cell_df.loc[ind, "complexity"] = rg[0].perimeter / (np.pi * rg[0].equivalent_diameter_area)
                cell_df.loc[ind, "area"] = rg[0].area
                cell_df.loc[ind, "solidity"] = rg[0].solidity
                try:
                    cell_df.loc[ind, "eccentricity"] = rg[0].axis_major_length / rg[0].axis_minor_length
                except:
                    cell_df.loc[ind, "eccentricity"] = np.nan

                cell_df.loc[ind, "use_flag"] = 1
            else:
                cell_df.loc[ind, "use_flag"] = 0

            im_rs = resize(np.pad(im, 16), snip.shape, preserve_range=True)
            cell_df.loc[ind, "lcp_intensity"] = np.mean(im_rs[snip > 50])
            # cell_df.loc[ind, "intertia_tensor"] = [rg[0].inertia_tensor]
            # cell_df.loc[ind, "intertia_eigs"] = rg[0].inertia_tensor_eigvals
            cell_df.loc[ind, "circularity"] = rg[0].area / (rg[0].axis_major_length**2*np.pi / 4)

            # if time_id == 273:
            #     print("check")

        df_list.append(cell_df)

    shape_df = pd.concat(df_list, axis=0, ignore_index=True)
    shape_df = shape_df.drop_duplicates(subset=["track_id", "t"])

    shape_df.to_csv(os.path.join(tracking_directory, "cell_shape_df" + suffix + ".csv"), index=False)

if __name__ == '__main__':
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "240219_LCP1_93hpf_to_127hpf"  # "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw" #
    overwrite_flag = True
    config_name = "tracking_cell.txt"

    extract_cell_shape_stats(root, project_name, config_name,  motion_orient_flag=True, shape_orient_flag=False)
