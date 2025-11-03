import os
import pandas as pd
import numpy as np
from joblib import dump, load
from pathlib import Path

def classify_cell_tracks(root: Path,
                         project_name: str,
                         tracking_config: str,
                         feature_df: pd.DataFrame,
                         classifier_path: Path,
                         tracking_range: tuple[int, int] | None = None
                         ):

    # --- open segmentation and experiment arrays ---
    tracking_root = root / "tracking" / project_name / tracking_config
    if tracking_range is not None:
        tracking_dir = tracking_root / f"{tracking_range[0]:04d}_{tracking_range[1]:04d}"
    else:
        tracking_results = sorted(tracking_root.glob("track*"))
        tracking_results = [d for d in tracking_results if d.is_dir()]
        if len(tracking_results) == 1:
            tracking_dir = tracking_results[0]
        elif len(tracking_results) == 0:
            raise FileNotFoundError(f"No tracking results found in {tracking_root}")
        else:
            raise ValueError(f"Multiple tracking results found in {tracking_root}, please specify tracking_range.")

    # load tracks, classifier, and nucleus features
    tracks_path = tracking_dir / "tracks.csv"
    tracks_df = pd.read_csv(tracks_path)
    clf = load(classifier_path)
    feature_path = tracking_dir / "mask_features.csv"
    feature_df = pd.read_csv(feature_path)

    # get predictions
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
    class_df_grouped = class_df_grouped.rename(
        columns={"logL0": "track_logL0", "logL1": "track_logL1", "logL2": "track_logL2"})

    class_df = class_df.merge(class_df_grouped, on="track_id", how="left")
    print("Check")


if __name__ == "__main__":
    
    # load zarr image file
    root = Path(r"Y:\killi_dynamics")
    project_name = "20251019_BC1-NLS_52-80hpf"
    tracking_config = "tracking_20251102"
    well_num = None
    n_workers = 1

    print("Loading tracking data for project:", project_name)
    # load mask features
    if well_num is None:
        tracks_path = root / "tracking" / project_name / tracking_config
    else:
        tracks_path = root / "tracking" / project_name / tracking_config / f"well{well_num:04}"

    # load classifier
    clf = load(classifier_path)

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