from src.qc import mask_qc_wrapper
from src.build_killi.fuse_masks import fuse_wrapper
from multiprocessing import freeze_support
from src.tracking import perform_tracking


if __name__ == '__main__':
    freeze_support()
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    # side 1
    # project_name = "20250419_BC1-NLSMSC_side1"
    # mask_qc_wrapper(root, project_name, par_flag=True, overwrite=True)
    #
    # # side 2
    # project_name = "20250419_BC1-NLSMSC_side2"
    # mask_qc_wrapper(root, project_name, par_flag=True, overwrite=True)

    # Fuse masks
    project_name = "20250419_BC1-NLSMSC"
    fuse_wrapper(root, project_prefix=project_name, par_flag=True, overwrite=False)

    # call tracking
    # project_name = "20250419_BC1-NLSMSC"
    # model_name = ""  # only used for ML segmentation
    # suffix = ""
    # config_name = "tracking_20250328_redux"
    # start_i = 0  # start frame for tracking
    # stop_i = None  # end frame for tracking
    #
    # perform_tracking(root, project_name, tracking_config=config_name, seg_model=model_name, well_num=None,
    #                  start_i=start_i, stop_i=stop_i, use_fused=True, suffix=suffix, par_seg_flag=True)