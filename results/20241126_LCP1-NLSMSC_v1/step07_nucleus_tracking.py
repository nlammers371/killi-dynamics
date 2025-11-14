from src.nucleus_dynamics.tracking.perform_tracking import perform_tracking


if __name__ == "__main__":

    # load zarr image file
    root = "E:\\Nick\\killi_immuno_paper\\"

    # call tracking
    project_name = "20241126_LCP1-NLSMSC"
    model_name = ""  # only used for ML segmentation
    suffix = ""
    config_name = "tracking_lcp_nuclei_v1"  # "tracking_lcp_long"
    start_i = 0  # start frame for tracking
    stop_i = None  # end frame for tracking

    perform_tracking(root, project_name, tracking_config=config_name, seg_model=model_name,
                     start_i=start_i, stop_i=stop_i,
                     use_fused=False, suffix=suffix, par_seg_flag=True)