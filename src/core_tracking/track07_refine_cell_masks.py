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


def refine_cell_masks(root, project_name, config_name, overwrite_flag=False, motion_orient_flag=True, shape_orient_flag=False):

    tracking_folder = config_name.replace(".txt", "")
    tracking_folder = tracking_folder.replace(".toml", "")

    tracking_directory = os.path.join(root, "built_data", "tracking", project_name, tracking_folder)
    snip_directory = os.path.join(root, "built_data", "shape_snips", project_name, tracking_folder, "class0", "")

    if motion_orient_flag:
        suffix = "_motion_oriented"
    elif shape_orient_flag:
        suffix = "_shape_oriented"
    else:
        suffix = ""

    snip_directory_refined = os.path.join(root, "built_data", "shape_snips_refined" + suffix, project_name, tracking_folder, "class0", "")
    if not os.path.isdir(snip_directory_refined):
        os.makedirs(snip_directory_refined)

    im_directory = os.path.join(root, "built_data", "shape_images", project_name, tracking_folder, "class0", "")
    im_directory_refined = os.path.join(root, "built_data", "image_snips_refined" + suffix, project_name,
                                          tracking_folder, "class0", "")
    if not os.path.isdir(im_directory_refined):
        os.makedirs(im_directory_refined)

    # load track and segment info
    # cfg = load_config(os.path.join(root, "metadata", project_name, config_name))
    # tracks_df, graph = to_tracks_layer(cfg)
    tracks_df = pd.read_csv(os.path.join(root, "built_data", "tracking", project_name, tracking_folder, "tracks_cleaned.csv"))
    # load sphere fit info
    sphere_df = pd.read_csv(os.path.join(root, "metadata", project_name, "sphere_df.csv"))

    metadata_file_path = os.path.join(root, "metadata", project_name, "metadata.json")
    f = open(metadata_file_path)
    metadata = json.load(f)
    scale_vec = np.asarray(
        [metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])

    print("loading cell shape masks...")
    track_index = np.unique(tracks_df["track_id"])

    for _, track_id in enumerate(tqdm(track_index)):

        # iterate through label masks
        cell_df = tracks_df.loc[tracks_df["track_id"] == track_id, ["track_id", "track_id_orig", "t", "z", "y", "x"]].copy()
        vel_cart_prev = np.nan
        for i, ind in enumerate(cell_df.index): #[180]]):
            track_id_orig = cell_df.loc[ind, "track_id_orig"]
            # extrack mask info from tracks
            time_id = cell_df.loc[ind, "t"]
            cell_time_ft = (cell_df["t"] >= time_id-1) & (cell_df["t"] <= time_id + 1)
            sphere_time_ft = (sphere_df["t"] >= time_id - 1) & (sphere_df["t"] <= time_id + 1)
            # centroid = frame_df.loc[ind, ["z", "y", "x"]].to_numpy()

            # get sphere center (we need this to convert velocity of sphereical coords)
            sphere_center_um = np.mean(sphere_df.loc[sphere_time_ft, ["Z", "Y", "X"]].to_numpy(), axis=0)
            sphere_center = np.divide(sphere_center_um, scale_vec)

            # get velocity vector
            v_df = cell_df.loc[cell_time_ft, ["z", "y", "x"]].to_numpy()
            centroid = cell_df.loc[cell_df["t"] == time_id, ["z", "y", "x"]].to_numpy().flatten()

            # make read/write name
            snip_name = f'snip_track{track_id_orig:04}_t{time_id:04}.jpg'
            snip_path = os.path.join(snip_directory, snip_name)
            snip_path_out = os.path.join(snip_directory_refined, snip_name)

            if (not os.path.exists(snip_path_out)) | overwrite_flag:
                im_name = f'im_track{track_id_orig:04}_t{time_id:04}.jpg'
                im_path = os.path.join(im_directory, im_name)
                im_path_out = os.path.join(im_directory_refined, im_name)

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
                    mask_new = np.zeros(im.shape, dtype=np.uint8)
                    lb_mask = label(mask_rs)
                    if np.max(lb_mask) > 1:
                        rg = regionprops(lb_mask)
                        area_vec = np.asarray([r["area"] for r in rg])
                        keep_i = rg[np.argmax(area_vec)].label
                    else:
                        keep_i = 1
                    mask_new[lb_mask == keep_i] = 255
                    mask_new = skimage.morphology.area_closing(mask_new)

                # reorient mask relative to cell movement direction
                if motion_orient_flag:
                    n_rows = v_df.size // 3
                    v_df = np.reshape(v_df, (n_rows, 3))
                    if v_df.shape[0] > 1:
                        v_df_comp = np.concatenate(([v_df[-2, :]], [v_df[-1, :]]), axis=0)
                    else:
                        v_df_comp = np.concatenate(([v_df[-1, :]], [v_df[-1, :]]), axis=0)
                    static_flag = len(np.unique(v_df_comp)) <= 3
                    if (not static_flag) | ~np.all(np.isnan(vel_cart_prev)):
                        # get normal vector
                        normal_vec = centroid - sphere_center
                        norm_u = normal_vec / np.sqrt(np.sum(normal_vec ** 2))

                        # get coordinates for points within the cell
                        # raw_lbs = np.unique(raw_lb_vec[ultrack_lb_vec == track_id])
                        # lb_mask = np.asarray([raw_lb_vec[i] in raw_lbs for i in range(len(raw_lb_vec))])
                        # zyxp = zyx_arr[lb_mask, :]

                        # find d for perp plane
                        d = np.sum(np.multiply(norm_u, centroid))

                        # convert centroid to spherical
                        centroid_sph = cartesian_to_spherical(centroid[2], centroid[1], centroid[0])
                        new_point_sph = np.asarray([c.value for c in centroid_sph])
                        new_point_sph[1] = new_point_sph[1] + 0.1  # increment in phi direction

                        # convert back to cartesian
                        new_point_cart = spherical_to_cartesian(new_point_sph[0], new_point_sph[1], new_point_sph[2])
                        new_point = np.asarray([c.value for c in new_point_cart])
                        new_point = new_point[::-1]
                        plane_point = new_point.copy()
                        plane_point[0] = (-norm_u[1] * plane_point[1] - norm_u[2] * plane_point[2] + d) / norm_u[0]

                        # calculate unit vectors to define my two snip axes
                        v1 = centroid - plane_point
                        v1 = v1 / np.sqrt(np.sum(v1 ** 2))
                        v2 = np.cross(v1, norm_u)
                        v2 = v2 / np.sqrt(np.sum(v2 ** 2))

                        # get velocity vector
                        # v_df_comp = v_df_comp #- sphere_center
                        # get start and end positions in spherical coords
                        if not static_flag:
                            start_cart = v_df_comp[-2, :]
                            end_cart = v_df_comp[-1, :]
                            # start_sph = cartesian_to_spherical(v_df_comp[-2, 2], v_df_comp[-2, 1], v_df_comp[-2, 0])
                            # start_sph = np.asarray([c.value for c in start_sph])
                            # end_sph = cartesian_to_spherical(v_df_comp[-1, 2], v_df_comp[-1, 1], v_df_comp[-1, 0])
                            # end_sph = np.asarray([c.value for c in end_sph])

                            # get velocity in cartesian coordinates
                            velocity_cart = end_cart - start_cart
                        else:
                            velocity_cart = vel_cart_prev
                        vel_u = velocity_cart / np.sqrt(np.sum(velocity_cart**2))

                        # take dot product with normal vector
                        vn_dot = np.dot(norm_u, vel_u)
                        vel_normal = vn_dot * norm_u
                        vel_planar = vel_u - vel_normal
                        vel_planar = vel_planar / np.sqrt(np.sum(vel_planar**2))

                        # calculate angle between plane velocity and x axis (v2)
                        dot1 = np.dot(vel_planar, v1)
                        dot2 = np.dot(vel_planar, v2)
                        rt_angle = np.arccos(dot2) / np.pi * 180

                        # rt_angle = rt_angle
                        if dot1 < 0:
                            rt_angle = -rt_angle
                        # else:
                        #     rt_angle = rt_angle[0]
                        im_rotated = skimage.transform.rotate(im, rt_angle, preserve_range=True).astype(np.uint8)
                        mask_rotated = skimage.transform.rotate(mask_new, rt_angle, preserve_range=True, order=1)
                        mask_rotated = (np.round(mask_rotated / 110) * 255).astype(np.uint8)  # be a little generous to ensure that we preserve connectedness

                        # update velocity
                        vel_cart_prev = velocity_cart
                    else:
                        regions = skimage.measure.regionprops(mask_new)
                        orientation = regions[0].orientation

                        # Rotate the image to align with the x-axis
                        angle_to_align_x = np.degrees(-orientation) - 90
                        mask_rotated = skimage.transform.rotate(mask_new, angle_to_align_x, preserve_range=True,
                                                                order=1)
                        mask_rotated = (np.round(mask_rotated / 110) * 255).astype(np.uint8)
                        im_rotated = skimage.transform.rotate(im, angle_to_align_x, preserve_range=True,
                                                                order=1).astype(np.uint8)

                elif shape_orient_flag:
                    regions = skimage.measure.regionprops(mask_new)
                    orientation = regions[0].orientation

                    # Rotate the image to align with the x-axis
                    angle_to_align_x = np.degrees(-orientation) - 90
                    mask_rotated = skimage.transform.rotate(mask_new, angle_to_align_x, preserve_range=True, order=1)
                    mask_rotated = (np.round(mask_rotated / 110) * 255).astype(np.uint8)
                    im_rotated = skimage.transform.rotate(im, angle_to_align_x, preserve_range=True,
                                                          order=1).astype(np.uint8)

                else:
                    im_rotated = im
                    mask_rotated = mask_new



                # if time_id == 273:
                #     print("check")

                # if time_id == 38:
                #     print("check")
                io.imsave(snip_path_out, mask_rotated, check_contrast=False)
                io.imsave(im_path_out, im_rotated, check_contrast=False)

        vel_cart_prev = np.nan

if __name__ == '__main__':

    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "240219_LCP1_93hpf_to_127hpf"  # "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw" #
    overwrite_flag = True
    config_name = "tracking_cell.txt"
    refine_cell_masks(root, project_name, config_name, overwrite_flag=overwrite_flag,
                      motion_orient_flag=True,
                      shape_orient_flag=False)