import os
from src.build_killi.nucleus_classification import build_mask_feature_wrapper
from glob2 import glob
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    # script to build features from mask zarr file
    # At this point, should have tracked all relevant experiments

    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    project_name = "20250311_LCP1-NLSMSC"
    tracking_config = "tracking_20250328_redux"
    tracking_range = [0, 2339]
    suffix = "_cb"
    par_flag = True
    well_num = 0

    read_path = os.path.join(root, "tracking", project_name, tracking_config, f"well{well_num:04}", "")

    # load mask feature df
    feature_df = pd.read_csv(read_path + "mask_features.csv")

    # load and combine frames
    df_list = sorted(glob(os.path.join(temp_dir, "features_*.csv")))
    feature_df_list = []
    for f in tqdm(df_list, "Loading feature files", unit="file"):
        df = pd.read_csv(f)
        feature_df_list.append(df)
    # concatenate
    feature_df = pd.concat(feature_df_list, ignore_index=True)
    feature_df.to_csv(save_path + "mask_features.csv", index=False)