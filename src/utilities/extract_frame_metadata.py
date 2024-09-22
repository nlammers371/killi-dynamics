from tqdm import tqdm
import glob2 as glob
import os
from typing import Any
from typing import Dict
from src.utilities.functions import path_leaf
import pandas as pd
from aicsimageio import AICSImage
import numpy as np


def parse_czi_metadata(czi_path, image_list, dT):

    imObject = AICSImage(czi_path)
    im_raw_dask = imObject.data
    im_shape = im_raw_dask.shape
    n_z_slices = im_shape[2]
    n_time_points = im_shape[0]
    # n_wells = 1

    scale_vec = np.asarray(imObject.physical_pixel_sizes)

    # extract frame times
    n_frames_total = len(image_list)
    frame_time_vec = np.arange(n_frames_total)*dT

    # # check for common nd2 artifact where time stamps jump midway through
    # dt_frame_approx = (imObject.frame_metadata(n_z_slices).channels[0].time.relativeTimeMs -
    #                    imObject.frame_metadata(0).channels[0].time.relativeTimeMs) / 1000
    # jump_ind = np.where(np.diff(frame_time_vec) > 2 * dt_frame_approx)[0]  # typically it is multiple orders of magnitude to large
    # if len(jump_ind) > 0:
    #     jump_ind = jump_ind[0]
    #     # prior to this point we will just use the time stamps. We will extrapolate to get subsequent time points
    #     nf = jump_ind - 1 - int(jump_ind / 2)
    #     dt_frame_est = (frame_time_vec[jump_ind - 1] - frame_time_vec[int(jump_ind / 2)]) / nf
    #     base_time = frame_time_vec[jump_ind - 1]
    #     for f in range(jump_ind, len(frame_time_vec)):
    #         frame_time_vec[f] = base_time + dt_frame_est * (f - jump_ind)
    # frame_time_vec = np.asarray(frame_time_vec)

    # get well positions
    # stage_zyx_array = np.empty((n_wells * n_time_points, 3))
    # for t in range(n_time_points):
    #     for w in range(n_wells):
    #         base_ind = t * n_wells + w
    #         slice_ind = base_ind * n_z_slices
    #
    #         stage_zyx_array[base_ind, :] = np.asarray(
    #             imObject.frame_metadata(slice_ind).channels[0].position.stagePositionUm)[::-1]

    ###################
    # Pull it together into dataframe
    ###################
    well_df = pd.DataFrame(np.arange(n_time_points)[:, np.newaxis], columns=["time_index"])
    # well_df["time_index"] = np.repeat(range(n_time_points), n_wells)
    # well_df["well_index"] = np.tile(range(n_wells), n_time_points)

    # add additional info
    well_df["time"] = frame_time_vec
    # well_df["stage_z_um"] = stage_zyx_array[:, 0]
    # well_df["stage_y_um"] = stage_zyx_array[:, 1]
    # well_df["stage_x_um"] = stage_zyx_array[:, 2]

    well_df["x_res_um_raw"] = scale_vec[0]
    well_df["y_res_um_raw"] = scale_vec[1]
    well_df["z_res_um_raw"] = scale_vec[2]

    # imObject.close()

    return well_df
# def parse_curation_metadata(root, experiment_date):
#     curation_path = os.path.join(root, "metadata", "curation", experiment_date + "_curation_info.xlsx")
#     if os.path.isfile(curation_path):
#         curation_xl = pd.ExcelFile(curation_path)
#         curation_df = curation_xl.parse(curation_xl.sheet_names[0])
#         curation_df_long = pd.melt(curation_df,
#                                    id_vars=["series_number", "notes", "example_flag", "follow_up_flag"],
#                                    var_name="time_string", value_name="qc_flag")
#         time_ind_vec = [int(t[1:]) for t in curation_df_long["time_string"].values]
#         curation_df_long["time_index"] = time_ind_vec
#         curation_df_long = curation_df_long.rename(columns={"series_number": "nd2_series"})
#
#     else:
#         curation_df_long = None
#         curation_df = None
#
#     return curation_df_long, curation_df

def extract_frame_metadata(
    root: str,
    experiment_date: str
) -> Dict[str, Any]:


    raw_directory = os.path.join(root, "raw_data", experiment_date, '')

    save_directory = os.path.join(root, "metadata", "frame_metadata", '')
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    # get list of images
    image_list = sorted(glob.glob(raw_directory + "*.nd2"))

    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    if len(image_list) > 1:
        raise Exception("Multiple .nd2 files were found in target directory. Make sure to put fullembryo images into a subdirectory")

    czi_path = image_list[0]
    imObject = nd2.ND2File(czi_path)
    im_array_dask = imObject.to_dask()
    nd2_shape = im_array_dask.shape

    metadata = dict({})
    metadata["n_time_points"] = nd2_shape[0]
    metadata["n_wells"] = nd2_shape[1]
    n_z = nd2_shape[2]
    if len(nd2_shape) == 6:
        n_channels = nd2_shape[3]
        n_x = nd2_shape[5]
        n_y = nd2_shape[4]
    else:
        n_channels = 1
        n_x = nd2_shape[4]
        n_y = nd2_shape[3]

    metadata["n_channels"] = n_channels
    metadata["zyx_shape"] = tuple([n_z, n_y, n_x])
    metadata["voxel_size_um"] = tuple(np.asarray(imObject.voxel_size())[::-1])

    im_name = path_leaf(czi_path)
    print("processing " + im_name)

    ####################
    # Process information from plate map
    # plate_df = parse_plate_metadata(root, experiment_date)

    ####################
    # Process extract information from nd2 metadata
    ####################
    # join on plate info using series id
    well_df = parse_czi_metadata(czi_path, image_list, dT)
    plate_cols = plate_df.columns
    well_cols = well_df.columns
    well_df = well_df.merge(plate_df, on="nd2_series", how="left")

    # reorder columns
    col_union = plate_cols.tolist() + well_cols.tolist()
    col_u = []
    [col_u.append(col) for col in col_union if col not in col_u]
    well_df = well_df.loc[:, col_u]

    ################
    # Finally, add curation info
    curation_df_long, curation_df_wide = parse_curation_metadata(root, experiment_date)
    if curation_df_long is not None:
        well_df = well_df.merge(curation_df_long, on=["nd2_series", "time_index"], how="left")
    well_df["estimated_stage_hpf"] = well_df["start_age_hpf"] + well_df["time"]/3600

    # save
    well_df.to_csv(os.path.join(root, "metadata", experiment_date + "_master_metadata_df.csv"), index=False)

    return metadata

if __name__ == "__main__":

    # set path to CellPose model to use
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\pecfin_dynamics\\fin_morphodynamics\\"
    experiment_date = "20240223"


    extract_frame_metadata(root=root, experiment_date=experiment_date)
