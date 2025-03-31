import napari
import os
from tqdm.contrib.concurrent import process_map
from functools import partial
import pandas as pd
import scipy.interpolate
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
from skimage.morphology import binary_closing, disk, binary_dilation, ball
import json
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

# # set parameters
def mask_iterator(t, raw_masks, segments, raw_data, sphere_df, tracks_df, scale_vec, im_directory,
                  save_directory, ds_factor, snip_dim, snip_dim_i, z_ref_snip_i, y_ref_snip_i, x_ref_snip_i, max_ref_i,
                  overwrite_flag):

    # for t in tqdm(range(segments.shape[0])):
    # raw_mask_arr = io.imread(mask_list[t])
    raw_mask_arr = np.asarray(raw_masks[t])

    # iterate through label masks
    frame_df = tracks_df.loc[tracks_df["t"] == t, :]
    frame_arr = np.asarray(segments[t])
    raw_data_arr = np.asarray(raw_data[t])

    # raw_data_arr = resize(raw_data_arr, raw_mask_arr.shape, preserve_range=True)
    sphere_center_um = sphere_df.loc[sphere_df["t"] == t, ["Z", "Y", "X"]].to_numpy()[0]
    sphere_center = np.divide(sphere_center_um, scale_vec)


    z_ref, y_ref, x_ref = np.meshgrid(range(frame_arr.shape[0]),
                                      range(frame_arr.shape[1]),
                                      range(frame_arr.shape[2]),
                                      indexing="ij")

        # z_ref_i, y_ref_i, x_ref_i = np.meshgrid(range(raw_data_arr.shape[0]),
        #                                   range(raw_data_arr.shape[1]),
        #                                   range(raw_data_arr.shape[2]),
        #                                   indexing="ij")

    foreground_mask = (frame_arr > 0) | (raw_mask_arr > 0)
    ultrack_lb_vec = frame_arr[foreground_mask]
    raw_lb_vec = raw_mask_arr[foreground_mask]
    # raw_data_vec = raw_data_arr[foreground_mask]
    zyx_arr = np.concatenate((np.reshape(z_ref[foreground_mask], (len(ultrack_lb_vec), 1)),
                             np.reshape(y_ref[foreground_mask], (len(ultrack_lb_vec), 1)),
                             np.reshape(x_ref[foreground_mask], (len(ultrack_lb_vec), 1))), axis=1)

    # define morph element to expand mask
    bp = ball(1)
    for i, ind in enumerate(frame_df.index):
        # extrack mask info from tracks
        track_id = frame_df.loc[ind, "track_id"]
        centroid = frame_df.loc[ind, ["z", "y", "x"]].to_numpy()

        # make read/write name
        snip_name = f'snip_track{track_id:04}_t{t:04}.jpg'
        snip_path = os.path.join(save_directory, snip_name)

        im_name = f'im_track{track_id:04}_t{t:04}.jpg'
        im_path = os.path.join(im_directory, im_name)

        if (not os.path.isfile(im_path)) | overwrite_flag:

            # get normal vector
            normal_vec = centroid - sphere_center
            norm_u = normal_vec / np.sqrt(np.sum(normal_vec**2))

            # get coordinates for points within the cell
            raw_lbs = np.unique(raw_lb_vec[ultrack_lb_vec == track_id])
            lb_mask = np.asarray([raw_lb_vec[i] in raw_lbs for i in range(len(raw_lb_vec))])
            zyxp = zyx_arr[lb_mask, :]

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
            plane_point[0] = (-norm_u[1]*plane_point[1] - norm_u[2]*plane_point[2] + d) / norm_u[0]

            # calculate unit vectors to define my two snip axes
            v1 = centroid - plane_point
            v1 = v1 / np.sqrt(np.sum(v1**2))
            v2 = np.cross(v1, norm_u)
            v2 = v2 / np.sqrt(np.sum(v2 ** 2))

            # take dot product to convert to plane coordinates
            zyxp_plane = zyxp - centroid
            c1 = np.round(np.sum(np.multiply(v1[np.newaxis], zyxp_plane), axis=1)).astype(np.uint8) + snip_dim // 2 + 1
            c2 = np.round(np.sum(np.multiply(v2[np.newaxis], zyxp_plane), axis=1)).astype(np.uint8) + snip_dim // 2 + 1

            keep_flags = (c1 < snip_dim) & (c2 < snip_dim)
            c1 = c1[keep_flags]
            c2 = c2[keep_flags]
            # convert to snip
            snip_raw = np.zeros((snip_dim, snip_dim)).astype(np.uint8)
            snip_raw[c1, c2] = 1
            # snip_dist = distance_transform_edt(snip_raw != 1)
            # snip_dist[snip_dist > 1] = 1
            morph_fp = disk(2)
            snip = (binary_closing(snip_raw > 0, morph_fp) * 255).astype(np.uint8)
            io.imsave(snip_path, snip, check_contrast=False)

            # grab mask and image blocks
            start_vec = np.min(zyxp, axis=0) - 1
            start_vec[start_vec < 0] = 0
            start_vec_i = start_vec * ds_factor
            stop_vec = np.max(zyxp, axis=0) + 1
            # stop_vec = np.asarray([stop_vec[i] if stop_vec[i]])
            stop_vec_i = stop_vec * ds_factor
            mask_block = raw_mask_arr[start_vec[0]:stop_vec[0], start_vec[1]:stop_vec[1], start_vec[2]:stop_vec[2]]
            image_block = raw_data_arr[start_vec_i[0]:stop_vec_i[0], start_vec_i[1]:stop_vec_i[1],
                          start_vec_i[2]:stop_vec_i[2]]
            mask_block_rs = binary_dilation(resize(mask_block, image_block.shape, preserve_range=True, order=0), bp)
            m_filter = mask_block_rs.flatten() > 0

            # get reference arrays
            z_ref_i, y_ref_i, x_ref_i = np.meshgrid(range(start_vec_i[0], stop_vec_i[0]),
                                                    range(start_vec_i[1], stop_vec_i[1]),
                                                    range(start_vec_i[2], stop_vec_i[2]),
                                                    indexing="ij")

            zyx_arr_i = np.concatenate((np.reshape(z_ref_i, (z_ref_i.size, 1)),
                                        np.reshape(y_ref_i, (y_ref_i.size, 1)),
                                        np.reshape(x_ref_i, (x_ref_i.size, 1))), axis=1)

            # calculate distance perpendicular to plane
            new_z = np.sum(np.multiply(norm_u[np.newaxis, :], zyx_arr_i), axis=1) - d
            new_z = new_z - np.min(new_z)
            new_y = np.sum(np.multiply(v1[np.newaxis, :], zyx_arr_i), axis=1)
            new_y = new_y - np.mean(new_y) + snip_dim_i // 2 + 1
            new_x = np.sum(np.multiply(v2[np.newaxis, :], zyx_arr_i), axis=1)
            new_x = new_x - np.mean(new_x) + snip_dim_i // 2 + 1

            # calculate the angle wrpt z axis
            new_zyx = np.concatenate((new_z[m_filter, np.newaxis], new_y[m_filter, np.newaxis], new_x[m_filter, np.newaxis]), axis=1)
            val_vec = np.reshape(image_block, (image_block.size, ))
            val_vec_ft = val_vec[m_filter]
            bck_val = np.median(val_vec[~m_filter])
            interp3 = scipy.interpolate.LinearNDInterpolator(new_zyx, val_vec_ft, fill_value=bck_val)
            i_interp = interp3(z_ref_snip_i, y_ref_snip_i, x_ref_snip_i)

            # max project
            i_max = np.max(i_interp, axis=0)
            i_max = np.round(i_max / max_ref_i * 255).astype(np.uint8)

            # save
            io.imsave(im_path, i_max, check_contrast=False)

def extract_cell_masks(root, project_name, config_name, par_flag=False, overwrite_flag=False, snip_dim=64, snip_dim_i=96, ds_factor=2,
                       max_ref_i=2500, cell_depth=50):

    tracking_folder = config_name.replace(".txt", "")
    tracking_folder = tracking_folder.replace(".toml", "")
    # image_dir = os.path.join(root, "built_data", project_name, "")

    print("loading track and mask info...")
    read_directory = os.path.join(root, "built_data", "tracking", project_name, tracking_folder)
    save_directory = os.path.join(root, "built_data", "shape_snips_or",  project_name, tracking_folder, "class0")
    im_directory = os.path.join(root, "built_data", "shape_images_or", project_name, tracking_folder, "class0")
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)
    if not os.path.isdir(im_directory):
        os.makedirs(im_directory)

    metadata_file_path = os.path.join(root, "metadata", project_name, "metadata.json")
    f = open(metadata_file_path)
    metadata = json.load(f)
    scale_vec = np.asarray([metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])

    # make snip ref grids
    z_ref_snip_i, y_ref_snip_i, x_ref_snip_i = np.meshgrid(range(cell_depth),
                                                        range(snip_dim_i),
                                                        range(snip_dim_i),
                                          indexing="ij")
    # zyx_arr_snip_i = np.concatenate((np.reshape(z_ref_snip_i, (z_ref_snip_i.size, 1)),
    #                                  np.reshape(y_ref_snip_i, (y_ref_snip_i.size, 1)),
    #                                  np.reshape(x_ref_snip_i, (x_ref_snip_i.size, 1))), axis=1)

    morph_fp = disk(2)
    # load track and segment info
    # cfg = load_config(os.path.join(root, "metadata", project_name, config_name))
    # tracks_df, graph = to_tracks_layer(cfg)
    tracks_df = pd.read_csv(os.path.join(read_directory, "tracks.csv"))
    segments = zarr.open(os.path.join(read_directory, "segments.zarr"), mode='r')

    # path to raw cell labels
    raw_mask_zarr = os.path.join(root, "built_data", "cleaned_cell_labels", project_name + ".zarr")
    raw_masks = zarr.open(raw_mask_zarr, mode='r')
    # mask_list = sorted(glob.glob(raw_mask_directory + "*.tif"))

    # path to raw image data
    raw_data_zarr = os.path.join(root, "built_data", "exported_image_files", project_name + ".zarr")
    raw_data = zarr.open(raw_data_zarr, mode='r')
    # load sphere fit info
    sphere_df = pd.read_csv(os.path.join(root, "metadata", project_name, "sphere_df.csv"))

    print("generating 2D cell profiles...")
    if par_flag:
        process_map(partial(mask_iterator, raw_masks=raw_masks, segments=segments, raw_data=raw_data, sphere_df=sphere_df,
                            tracks_df=tracks_df, scale_vec=scale_vec, im_directory=im_directory,
                            save_directory=save_directory, ds_factor=ds_factor, snip_dim=snip_dim, snip_dim_i=snip_dim_i,
                            z_ref_snip_i=z_ref_snip_i, y_ref_snip_i=y_ref_snip_i, x_ref_snip_i=x_ref_snip_i,
                            max_ref_i=max_ref_i, overwrite_flag=overwrite_flag),
                    range(segments.shape[0]), max_workers=10)
    else:

        for t in tqdm(range(259, segments.shape[0])):
            mask_iterator(t, raw_masks=raw_masks, segments=segments, raw_data=raw_data, sphere_df=sphere_df,
                    tracks_df=tracks_df, scale_vec=scale_vec, im_directory=im_directory,
                    save_directory=save_directory, ds_factor=ds_factor, snip_dim=snip_dim, snip_dim_i=snip_dim_i,
                    z_ref_snip_i=z_ref_snip_i, y_ref_snip_i=y_ref_snip_i, x_ref_snip_i=x_ref_snip_i,
                    max_ref_i=max_ref_i, overwrite_flag=overwrite_flag)

    return {}



if __name__ == '__main__':
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "240219_LCP1_93hpf_to_127hpf" #"231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw" #

    snip_dim = 64
    overwrite_flag = False
    config_name = "tracking_cell.txt" #"tracking_v17.txt"

    extract_cell_masks(root, project_name, config_name, par_flag=True, overwrite_flag=overwrite_flag)

    # process_map(
    #     partial(extract_cell_masks, image_list=image_list, project_path=project_path,
    #             overwrite_flag=overwrite_flag, metadata_file_path=metadata_file_path,
    #             file_prefix=file_prefix, tres=tres, resampling_scale=resampling_scale),
    #     range(len(image_list)), max_workers=n_workers)
