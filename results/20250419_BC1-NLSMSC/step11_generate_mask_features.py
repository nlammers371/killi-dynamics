import os
from src.build_killi.nucleus_classification import build_mask_feature_wrapper
from glob2 import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    # script to build features from mask zarr file
    # At this point, should have tracked all relevant experiments

    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20250419_BC1-NLSMSC"
    tracking_config = "tracking_20250328_redux"

    start_i = 0
    stop_i = 614
    tracking_range = [start_i, stop_i]
    suffix = ""
    scale_vec = np.asarray([3.0, 0.85, 0.85])
    well_num = 0
    par_flag = True

    project_path = os.path.join(root, "tracking", project_name, tracking_config, f"well{well_num:04}", "")
    # save_path = os.path.join(project_path, f"track_{tracking_range[0]:04}" + f"_{tracking_range[1]:04}" + suffix, "")
    # temp_dir = Path(save_path) / "temp_features"
    # read_path = os.path.join(root, "tracking", project_name, tracking_config, f"well{well_num:04}", "")

    build_mask_feature_wrapper(root, project_name, tracking_config, well_num=0,
                               start_i=tracking_range[0], stop_i=tracking_range[1], par_flag=par_flag,
                               overwrite=False, suffix="", nls_channel=None, n_workers=None)