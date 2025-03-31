from src.nucleus_dynamics.tracking.perform_tracking import perform_tracking, check_tracking
import napari
import numpy as np

if __name__ == "__main__":

    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    # call tracking
    project_name = "20250311_LCP1-NLSMSC"
    model_name = ""  # only used for ML segmentation
    config_name = "tracking_20250328_redux"

    start_i = 0
    stop_i = 2339
    view_range = np.arange(2137, 2148)
    viewer = check_tracking(root, project_name, tracking_config=config_name, seg_model=model_name, well_num=None,
                            start_i=start_i, stop_i=stop_i, view_range=view_range, use_fused=True, suffix="", )

    napari.run()

    print("Check")