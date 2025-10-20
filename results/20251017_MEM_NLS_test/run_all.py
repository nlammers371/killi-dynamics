from src.data_io.czi_export import export_czi_to_zarr
import numpy as np
from multiprocessing import freeze_support
from src.segmentation.li_thresholding import calculate_li_trend, estimate_li_thresh
from src.segmentation.segmentation_wrappers import segment_nuclei_thresh
from src.qc import mask_qc_wrapper
from pathlib import Path

if __name__ == '__main__':
    #----------------------------
    # Export raw CZI timelapse to OME-Zarr

    freeze_support()

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([3, 0.85, 0.85])
    last_i = None

    # # set path parameters
    # raw_data_root = r"E:\pipeline_dev\killi_dynamics"
    #
    # file_prefix_vec = ["bmm_utr-nlsmsc_subset_210um"]
    #
    # channel_names = ["mem-GFP", "nls-mScar"]
    # keep_channel_vec = [True, True]
    #
    # # Specify the path to the output OME-Zarr file and metadata file
    # save_root = r"E:\pipeline_dev\killi_dynamics"
    # project_name_vec = ["MEM_NLS_test"]
    #
    # overwrite = True
    #
    # for i in range(len(project_name_vec)):
    #     file_prefix = file_prefix_vec[i]
    #     project_name = project_name_vec[i]
    #     export_czi_to_zarr(raw_data_root=raw_data_root,
    #                        file_prefix=file_prefix,
    #                        project_name=project_name, save_root=save_root,
    #                        resampling_scale=resampling_scale,
    #                        n_workers=8,
    #                        channel_names=channel_names,
    #                        channels_to_keep=keep_channel_vec,
    #                        overwrite_flag=overwrite, last_i=last_i)
    #
    #
    # #----------------------------
    # # Estimate LI threshold trend
    root = Path(r"E:\pipeline_dev\killi_dynamics")
    project_name = "MEM_NLS_test"
    # estimate_li_thresh(root, project_name, interval=3, nuclear_channel=None, last_i=None, timeout=60 * 10)
    #
    # # get trend
    # calculate_li_trend(root, project_prefix=project_name)
    #
    # # ----------------------------
    # # Segment nuclei using LI thresholding
    # segment_nuclei_thresh(root, project_name, overwrite=True, n_workers=8)
    #
    # mask_qc_wrapper(root, project_name, mask_type="li_segmentation", n_workers=1, overwrite=True)