import os
from src.build_killi.nucleus_classification import build_mask_feature_wrapper
from glob2 import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
from joblib import dump, load

if __name__ == "__main__":
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

    print("Loading tracking data for project:", project_name)
    # load mask features
    project_path = os.path.join(root, "tracking", project_name, tracking_config, f"well{well_num:04}", "")
    save_path = os.path.join(project_path, f"track_{tracking_range[0]:04}" + f"_{tracking_range[1]:04}" + suffix, "")
    feature_df = pd.read_csv(save_path + "mask_features.csv")

    # load classifier
    classifier_path = track_path = os.path.join(root, "tracking", project_name, "nucleus_classifier", "")
    clf = load(os.path.join(classifier_path, "nucleus_rf_classifier.joblib"))

    print("Getting predictions....")
    # strip to just features for prediction
    X = feature_df.drop(columns=["frame", "label"], axis=1)
    Y = clf.predict(X)
    logL = clf.predict_log_proba(X)

    # get modal prediction for each track
    class_df = pd.DataFrame(feature_df.loc[:, ["frame", "label"]]).rename(columns={"label": "track_id", "frame": "t"})
    class_df["frame_class"] = Y
    class_df[["logL0", "logL1", "logL2"]] = logL
    class_df_grouped = class_df.loc[:, ["track_id", "logL0", "logL1", "logL2"]].groupby(
                                                                                    "track_id").mean().reset_index()
    class_df_grouped["track_class"] = np.argmax(class_df_grouped.loc[:, ["logL0", "logL1", "logL2"]], axis=1)
    class_df_grouped = class_df_grouped.rename(columns={"logL0":"track_logL0", "logL1":"track_logL1", "logL2":"track_logL2"})

    # now join back onto the full track
    class_df = class_df.merge(class_df_grouped, on="track_id", how="left")

    # save results
    class_df.to_csv(save_path + "track_class_df_full.csv", index=False)
    class_df_grouped.to_csv(save_path + "track_class_df.csv", index=False)