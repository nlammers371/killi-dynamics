import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.classify_nuclei.build_features import process_frame, build_tracked_mask_features
from src.classify_nuclei.classify_tracks import classify_cell_tracks
from src.fluorescence.get_mask_fluorescence import compute_mean_fluo_from_foreground
from src.fluorescence.extract_image_foreground import extract_foreground_intensities
from src.tracking.track_processing import smooth_tracks_wrapper, find_dropped_nuclei
from pathlib import Path

if __name__ == "__main__":
    # root = Path(r"Y:\killi_dynamics")
    root = Path("/media/nick/hdd011/killi_dynamics/")
    project_name = "20251019_BC1-NLS_52-80hpf"
    tracking_config = "tracking_20251102"
    flows_flag = False #True
    well_num = None
    n_workers = 1
    seg_type = "li_segmentation"
    mdl_path = Path(r"C:\Users\nlammers\Projects\killi-dynamics\src\classify_nuclei\models\nucleus_rf_classifier_v0.joblib")

    # first extract foreground intensities. Creates lightweight zarr arrays with only pixels within masks
    # extract_foreground_intensities(root, project_name, n_workers=n_workers, overwrite=False)
    compute_mean_fluo_from_foreground(root=root,
                                      project_name=project_name,
                                      tracking_config=tracking_config,
                                      n_workers=n_workers,
                                      overwrite=False)

    # stitch tracks?

    # process frames to build feature dataframe
    feature_df = build_tracked_mask_features(
                                root=root,
                                project_name=project_name,
                                seg_type=seg_type,
                                tracking_config=tracking_config,
                                well_num=well_num,
                                use_foreground=True,
                                used_optical_flow=flows_flag,
                                n_workers=n_workers,
                                mask_field="clean")

    classify_cell_tracks(
        root=root,
        project_name=project_name,
        tracking_config=tracking_config,
        used_optical_flow=flows_flag,
        classifier_path=mdl_path
    )

    # smooth tracks
    smoothed_tracks = smooth_tracks_wrapper(root=root,
                                            project_name=project_name,
                                            tracking_config=tracking_config,
                                            used_flow=flows_flag,
                                            n_workers=n_workers,
                                            overwrite=False,
                                            tracking_range=None)

    # ---------------------------
    # Attend to cells that were dropped during tracking
    # ---------------------------
    # add missed nuclei?
    find_dropped_nuclei(root=root,
                        project_name=project_name,
                        tracking_config=tracking_config,
                        used_optical_flow=flows_flag,
                        overwrite=False,
                        n_workers=n_workers)


    build_tracked_mask_features(
        root=root,
        project_name=project_name,
        seg_type=seg_type,
        tracking_config=tracking_config,
        well_num=well_num,
        use_foreground=True,
        used_optical_flow=flows_flag,
        n_workers=n_workers,
        mask_field="clean",
        process_dropped_nuclei=True)

    classify_cell_tracks(
        root=root,
        project_name=project_name,
        tracking_config=tracking_config,
        used_optical_flow=True,
        classifier_path=mdl_path,
        classify_dropped_nuclei=True,
    )
