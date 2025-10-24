from src.data_io.czi_export import export_czi_to_zarr
import numpy as np
from multiprocessing import freeze_support
from src.segmentation.li_thresholding import calculate_li_trend, estimate_li_thresh
from src.segmentation.segmentation_wrappers import segment_nuclei_thresh
from src.qc import mask_qc_wrapper
from src.geometry.geom_wrappers import fit_surf_sphere_trend, fit_sh_trend
from pathlib import Path

if __name__ == '__main__':
    freeze_support()

    # ----------------------------
    # Export czi -> zarr
    # ----------------------------

    resampling_scale = np.asarray([3, 0.85, 0.85])
    dT = 90  # time between frames in seconds
    last_i = None

    # set path parameters
    root = r"E:\pipeline_dev\killi_dynamics"

    file_prefix_vec = ["bmm_utr-nlsmsc_subset_210um"]

    channel_names = ["mem-GFP", "nls-mScar"]
    keep_channel_vec = [True, True]

    # Specify the path to the output OME-Zarr file and metadata file
    project_name = "MEM_NLS_test"

    overwrite = True
    export_czi_to_zarr(raw_data_root=root,
                       project_name=project_name,
                       resampling_scale=resampling_scale,
                       n_workers=8,
                       tres=dT,
                       channel_names=channel_names,
                       channels_to_keep=keep_channel_vec,
                       overwrite_flag=overwrite, last_i=last_i)

    # ----------------------------
    # Fuse image halves (2-sided experiments only)
    # ----------------------------


    # ----------------------------
    # Segmentation (LI threshold estimation)
    # ----------------------------

    estimate_li_thresh(root, project_name, interval=3, nuclear_channel=None, last_i=None, timeout=60 * 10)

    # get trend
    calculate_li_trend(root, project_prefix=project_name)

    # ----------------------------
    # Thresh-based segmentation
    # ----------------------------
    segment_nuclei_thresh(root, project_name, overwrite=True, n_workers=8)

    # ----------------------------
    # Perfom mask QC
    # ----------------------------
    mask_qc_wrapper(root, project_name, mask_type="li_segmentation", n_workers=1, overwrite=True)

    # ----------------------------
    # Fit sphere and SH surface to embryo
    # ----------------------------
    fit_surf_sphere_trend(root, project_name, seg_type="li_segmentation", well=None)
    fit_sh_trend(root, project_name, seg_type="li_segmentation", well=None, L_max=8, nside=16, sh_ridge=1e-6)

    # ----------------------------
    # Call live cell tracking
    # ----------------------------
    # from src.tracking.core_tracking import perform_tracking
    # from pathlib import Path
    #
    # config_name = "tracking_v0"
    # start_i = 0  # start frame for tracking
    # stop_i = None  # end frame for tracking
    #
    # perform_tracking(root, project_name, tracking_config=config_name, seg_model="li_segmentation", well_num=None,
    #                  start_i=start_i, stop_i=stop_i, suffix="", par_seg_flag=True, overwrite_tracking=True)