from src.build_killi.build_utils import integrate_fluorescence_wrapper, transfer_fluorescence_wrapper
import os
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20250419_BC1-NLSMSC"

    # call function
    print("Integrating fluorescence for project:", project_name)
    # integrate_fluorescence_wrapper(root, project_name, fluo_channel=0, par_flag=True, overwrite=True, n_workers=12)

    transfer_fluorescence_wrapper(root, project_name, fused_flag=True, tracking_range=[0, 614],
                                  tracking_config="tracking_20250328_redux", par_flag=True, overwrite=True)