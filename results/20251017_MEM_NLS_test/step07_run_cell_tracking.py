from src.tracking.core_tracking import perform_tracking
from pathlib import Path

if __name__ == "__main__":

    # load zarr image file
    root = Path(r"E:\pipeline_dev\killi_dynamics")

    # call tracking
    project_name = "MEM_NLS_test"
    model_name = ""  # only used for ML segmentation
    suffix = ""
    config_name = "tracking_v0"
    start_i = 0  # start frame for tracking
    stop_i = None  # end frame for tracking

    perform_tracking(root, project_name, tracking_config=config_name, seg_model=model_name, well_num=None,
                     start_i=start_i, stop_i=stop_i, use_fused=True, suffix=suffix, par_seg_flag=True)