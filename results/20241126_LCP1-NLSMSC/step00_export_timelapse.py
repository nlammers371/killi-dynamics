from src.data_io._Archive.czi_export import export_czi_to_zarr
import numpy as np
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([3, 0.8, 0.8])
    tres = 90  # time resolution in seconds
    last_i = None
    # set path parameters
    raw_data_root = "D:\\Syd\\241126_EXP62_LCP1_NLSMSC_96hpf_StabWound\\"
    file_prefix_vec = ["E3_TL_2024_11_26__21_56_34_174"]

    channel_names = ["lcp1", "nls"]

    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name_vec = ["20251126_LCP1-NLSMSC"]

    overwrite = False

    for i in range(len(project_name_vec)):
        file_prefix = file_prefix_vec[i]
        project_name = project_name_vec[i]
        export_czi_to_zarr(raw_data_root, file_prefix, project_name, save_root, tres, resampling_scale=resampling_scale,
                           par_flag=True, channel_names=channel_names,
                           channel_to_use=None, overwrite_flag=overwrite, last_i=last_i)