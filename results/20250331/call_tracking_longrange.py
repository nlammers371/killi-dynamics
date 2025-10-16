from src.nucleus_dynamics.tracking.perform_tracking import perform_tracking


if __name__ == "__main__":
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    # call tracking
    project_name = "20250311_LCP1-NLSMSC"
    model_name = ""  # only used for ML segmentation
    config_name = "tracking_20250328_longrange"
    start_i = 0  # 2000 # start frame for tracking
    stop_i = 2100  # 2100  # end frame for tracking

    perform_tracking(root, project_name, tracking_config=config_name, seg_model=model_name, well_num=None, start_i=start_i,
                     stop_i=stop_i, use_fused=True, suffix="", par_seg_flag=True)