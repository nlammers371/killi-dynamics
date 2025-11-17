import os
import pandas as pd
import numpy as np
from joblib import dump, load
from pathlib import Path
from src.tracking.track_processing import _load_tracks

def classify_cell_tracks(root: Path,
                         project_name: str,
                         tracking_config: str,
                         classifier_path: Path,
                         used_optical_flow: bool = True,
                         tracking_range: tuple[int, int] | None = None,
                         classify_dropped_nuclei: bool = False
                         ):


    if not classify_dropped_nuclei:
        tracks_df, tracking_dir = _load_tracks(root=root,
                                                project_name=project_name,
                                                tracking_config=tracking_config,
                                                tracking_range=tracking_range,
                                                prefer_flow=used_optical_flow)

        feature_path = tracking_dir / "mask_features.csv"
        feature_df = pd.read_csv(feature_path)
    else:
        # load dropped nuclei features
        _, tracking_dir = _load_tracks(root=root,
                                        project_name=project_name,
                                        tracking_config=tracking_config,
                                        tracking_range=tracking_range,
                                        prefer_flow=used_optical_flow)

        feature_path = tracking_dir / "dropped_nuclei_mask_features.csv"
        # tracks_df = pd.read_csv(tracking_dir / "dropped_nuclei.csv")
        feature_df = pd.read_csv(feature_path)


    clf = load(classifier_path)


    # get predictions
    X = feature_df.drop(columns=["frame", "label"], axis=1)
    Y = clf.predict(X)
    logL = clf.predict_log_proba(X)

    # get modal prediction for each track
    class_df = pd.DataFrame(feature_df.loc[:, ["frame", "label"]]).rename(columns={"label": "track_id", "frame": "t"})
    class_df["frame_class"] = Y
    class_df[["logL0", "logL1", "logL2"]] = logL

    # class_df = class_df.merge(class_df_grouped, on="track_id", how="left")

    # save results
    class_df["mdl_name"] = classifier_path.name

    if classify_dropped_nuclei:
        class_df.to_csv(tracking_dir / "dropped_nuclei_class_df.csv", index=False)
    else:
        class_df_grouped = class_df.loc[:, ["track_id", "logL0", "logL1", "logL2"]].groupby(
            "track_id").mean().reset_index()
        class_df_grouped["track_class"] = np.argmax(class_df_grouped.loc[:, ["logL0", "logL1", "logL2"]], axis=1)
        class_df_grouped = class_df_grouped.rename(
            columns={"logL0": "track_logL0", "logL1": "track_logL1", "logL2": "track_logL2"})
        class_df_grouped["mdl_name"] = classifier_path.name

        class_df.to_csv(tracking_dir / "cell_class_df.csv", index=False)
        class_df_grouped.to_csv(tracking_dir / "track_class_df.csv", index=False)

    return class_df

