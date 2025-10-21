from src.data_io.czi_export import export_czi_to_zarr
import numpy as np
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([3, 1, 1])
    tres = 124  # time resolution in seconds
    last_i = 805
    # set path parameters
    raw_data_root = "D:\\Syd\\241114_EXP60_LCP1_NLSMSC_72h_to_96h_DualSided\\"
    file_prefix_vec = ["E2_2024_11_14__20_21_18_968_G1", "E2_2024_11_14__20_21_18_968_G2"]

    channel_names = ["lcp1", "nls"]

    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name_vec = ["20241114_LCP1-NLSMSC_side1", "20241114_LCP1-NLSMSC_side2"]
    overwrite = True

    for i in range(1, len(project_name_vec)):
        file_prefix = file_prefix_vec[i]
        project_name = project_name_vec[i]
        export_czi_to_zarr(raw_data_root, file_prefix, project_name, save_root, tres, resampling_scale=resampling_scale,
                           par_flag=True, channel_names=channel_names,
                           channel_to_use=None, overwrite_flag=overwrite, last_i=last_i)