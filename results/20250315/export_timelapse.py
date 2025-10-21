from src.data_io._Archive.czi_export import export_czi_to_zarr
import numpy as np
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([3, 1, 1])
    tres = 91.63  # time resolution in seconds
    last_i = None
    # set path parameters
    raw_data_root = "D:\\Syd\\250311_NLSxLCP1_26h_to_96h_R2\\"
    file_prefix_vec = ["E1_TL_2025_03_11__21_46_23_363_G1", "E1_TL_2025_03_11__21_46_23_363_G2"]

    channel_names = ["lcp1", "nls"]

    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name_vec = ["20250311_LCP1-NLSMSC_side1", "20250311_LCP1-NLSMSC_side2"]

    overwrite = False

    for i in range(len(project_name_vec)):
        file_prefix = file_prefix_vec[i]
        project_name = project_name_vec[i]
        export_czi_to_zarr(raw_data_root, file_prefix, project_name, save_root, tres, resampling_scale=resampling_scale,
                           par_flag=True, channel_names=channel_names,
                           channel_to_use=None, overwrite_flag=overwrite, last_i=last_i)