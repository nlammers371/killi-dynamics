from src.classify_nuclei.build_features import process_frame, build_tracked_mask_features
# from src.classify_nuclei.classify_tracks import classify_nucleus_tracks
from pathlib import Path

if __name__ == "__main__":
    root = Path(r"Y:\killi_dynamics")
    project_name = "20251019_BC1-NLS_52-80hpf"
    tracking_config = "tracking_20251102"
    well_num = None
    n_workers = 12
    seg_type = "li_segmentation"

    # process frames to build feature dataframe
    feature_df = build_tracked_mask_features(
                                root=root,
                                project_name=project_name,
                                seg_type=seg_type,
                                tracking_config=tracking_config,
                                well_num=well_num,
                                use_foreground=True,
                                n_workers=n_workers,
                                mask_field="clean")