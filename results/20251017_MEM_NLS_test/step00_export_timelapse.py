from src.data_io.czi_export import export_czi_to_zarr
import numpy as np
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([3, 0.85, 0.85])
    last_i = None

    # set path parameters
    raw_data_root = r"E:\pipeline_dev\killi_dynamics"

    file_prefix_vec = ["bmm_utr-nlsmsc_subset_210um"]

    channel_names = ["mem-GFP", "nls-mScar"]
    keep_channel_vec = [True, True]

    # Specify the path to the output OME-Zarr file and metadata file
    save_root = r"E:\pipeline_dev\killi_dynamics"
    project_name_vec = ["MEM_NLS_test"]

    overwrite = True

    for i in range(len(project_name_vec)):
        file_prefix = file_prefix_vec[i]
        project_name = project_name_vec[i]
        export_czi_to_zarr(raw_data_root=raw_data_root,
                           file_prefix=file_prefix,
                           project_name=project_name, save_root=save_root,
                           resampling_scale=resampling_scale,
                           n_workers=8,
                           channel_names=channel_names,
                           channels_to_keep=keep_channel_vec,
                           overwrite_flag=overwrite, last_i=last_i)