from src.data_io._Archive.czi_export import export_czi_to_zarr
import numpy as np
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([3, 0.85, 0.85])
    tres = 90  # time resolution in seconds
    last_i = None
    # set path parameters
    raw_data_root = "F:\\Cooper\\250729_NLS-F_LCP1-M_23hpf_DualSided_90sInterval_3umZ_650umStack\\"
    file_prefix_vec = ["E1_TL_23hpf_2025_07_29__20_56_01_694_G1", "E1_TL_23hpf_2025_07_29__20_56_01_694_G2"]

    channel_names = ["lcp1", "nls"]
    keep_channel_vec = [False, True]
    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "I:\\Nick\\killi_tracker\\"
    project_name_vec = ["20250729_LCP1-NLSMSC_side1", "20250729_LCP1-NLSMSC_side2"]

    overwrite = False

    for i in range(len(project_name_vec)):
        file_prefix = file_prefix_vec[i]
        project_name = project_name_vec[i]
        export_czi_to_zarr(raw_data_root, file_prefix, project_name, save_root, tres, resampling_scale=resampling_scale,
                           par_flag=True, channel_names=channel_names, channels_to_keep=keep_channel_vec,
                           channel_to_use=None, overwrite_flag=overwrite, last_i=last_i)