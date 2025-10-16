from src.nucleus_dynamics.export_to_zarr.export_czi_to_zarr import export_czi_to_zarr
import numpy as np
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([3, 1, 1])
    tres = 124.59  # time resolution in seconds
    last_i = None
    # set path parameters
    raw_data_root = "D:\\Syd\\250116_NLS-LCP1_24-96_DualSided_120sInterval_3uMslice\\"
    file_prefix_vec = ["E3_TL_24-96_NLS-LCP1_2025_01_16__21_11_28_090_G1", "reset_E3_TL_24-96_NLS-LCP1_2025_01_17__01_15_08_719_G1",
                       "E3_TL_24-96_NLS-LCP1_2025_01_16__21_11_28_090_G2", "reset_E3_TL_24-96_NLS-LCP1_2025_01_17__01_15_08_719_G2"]

    channel_names = ["lcp1", "nls"]

    # Specify the path to the output OME-Zarr file and metadata file
    save_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name_vec = ["20250116_LCP1-NLSMSC_side1", "20250116_LCP1-NLSMSC_side1_reset",
                        "20250116_LCP1-NLSMSC_side2", "20250116_LCP1-NLSMSC_side2_reset"]

    overwrite = True

    for i in range(0, len(project_name_vec)):
        file_prefix = file_prefix_vec[i]
        project_name = project_name_vec[i]
        export_czi_to_zarr(raw_data_root, file_prefix, project_name, save_root, tres, resampling_scale=resampling_scale,
                           par_flag=True, channel_names=channel_names,
                           channel_to_use=None, overwrite_flag=overwrite, last_i=last_i)