from src.build_killi.build_utils import integrate_fluorescence_wrapper, transfer_fluorescence_wrapper
import os
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20250311_LCP1-NLSMSC"
    use_markers = True
    start_i = 1200
    stop_i = 2339
    tracking_range = [start_i, stop_i]
    tracking_config = "tracking_20250328_redux"  # only used for ML segmentation
    # call function
    print("Integrating fluorescence for project:", project_name)
    # integrate_fluorescence_wrapper(root, project_name, fluo_channel=0, par_flag=True, overwrite=True,
    #                                use_markers_flag=use_markers, n_workers=24, start_i=1200)

    # transfer fluorescence
    transfer_fluorescence_wrapper(root, project_name=project_name, tracking_config=tracking_config, use_markers_flag=use_markers,
                                  tracking_range=tracking_range, par_flag=True, n_workers=24, suffix="")