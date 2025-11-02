from src.tracking.core_tracking import perform_tracking
from pathlib import Path

if __name__ == "__main__":

    # call tracking
    root = Path(r"Y:\killi_dynamics")
    project_name = "20251019_BC1-NLS_52-80hpf"
    seg_type = "li_segmentation"
    config_name = "tracking_20251102"
    start_i = 0  # start frame for tracking
    stop_i = None  # end frame for tracking

    perform_tracking(root, project_name, seg_type=seg_type, tracking_config=config_name, well_num=None,
                     start_i=start_i, stop_i=stop_i, par_seg_flag=True, overwrite_tracking=True)