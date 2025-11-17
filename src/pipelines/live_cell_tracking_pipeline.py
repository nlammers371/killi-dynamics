"""Composable pipeline for lightsheet live-cell tracking datasets.

The helper below mirrors the ad-hoc scripts stored in ``results/20251027``
but exposes them as a single callable suitable for new projects. Each step
can be toggled on/off so callers may resume partially processed datasets
without re-running expensive stages.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import yaml

from src.classify_nuclei.build_features import build_tracked_mask_features
from src.classify_nuclei.classify_tracks import classify_cell_tracks
from src.fluorescence.extract_image_foreground import extract_foreground_intensities
from src.fluorescence.get_mask_fluorescence import compute_mean_fluo_from_foreground
from src.geometry.geom_wrappers import fit_surf_sphere_trend
from src.qc.mask_qc import mask_qc_wrapper
from src.segmentation.li_thresholding import run_li_threshold_pipeline
from src.segmentation.segmentation_wrappers import segment_nuclei_thresh
from src.symmetry_breaking.cell_cluster_tracking import (
    ClusterTrackingConfig,
    find_clusters_per_timepoint,
    stitch_tracklets,
    track_clusters_over_time,
)
from src.tracking.core_tracking import perform_tracking
from src.tracking.track_processing import find_dropped_nuclei, smooth_tracks_wrapper
from src.data_io.track_io import _load_track_data


def run_live_cell_tracking_pipeline(
    *,
    root: Path,
    project_name: str,
    seg_type: str = "li_segmentation",
    tracking_config: str = "tracking_default",
    classifier_path: Path | None = None,
    run_li_search: bool = False,
    run_segmentation: bool = False,
    run_qc: bool = True,
    run_tracking: bool = True,
    run_surface_fit: bool = True,
    run_fluorescence: bool = True,
    run_feature_build: bool = True,
    run_classification: bool = True,
    run_track_smoothing: bool = True,
    run_dropped_search: bool = False,
    run_cluster_tracking: bool = True,
    n_workers: int = 8,
    mask_field: str = "clean",
    use_optical_flow: bool = True,
    well_num: int | None = None,
    foreground_kwargs: Mapping[str, object] | None = None,
) -> None:
    """Execute the canonical live-cell tracking pipeline.

    Parameters
    ----------
    root:
        Project root that contains the ``built_data`` and output folders.
    project_name:
        Name of the dataset to process (matches Zarr directory names).
    seg_type:
        Segmentation key, e.g. ``"li_segmentation"`` or ``"cellpose"``.
    tracking_config:
        Ultrack configuration name (without extension).
    classifier_path:
        Optional path to a joblib classifier used for label prediction. If
        ``None``, classification is skipped even when ``run_classification``
        is ``True``.
    run_* flags:
        Toggle individual stages so the pipeline can resume from any step.
    n_workers:
        Shared worker count for parallelized steps.
    mask_field:
        Mask column used when building track-level features.
    use_optical_flow:
        Whether the tracking run leveraged optical flow (propagated downstream
        for smoothing/classification calls).
    well_num:
        Optional well number for multi-well acquisitions.
    foreground_kwargs:
        Extra keyword arguments forwarded to
        :func:`extract_foreground_intensities`.
    """

    root = Path(root)

    # 1) Segmentation search + execution
    if run_li_search:
        run_li_threshold_pipeline(root=root, project_name=project_name, use_subsampling=True)

    if run_segmentation:
        segment_nuclei_thresh(
            root=root,
            project_name=project_name,
            nuclear_channel=1,
            segment_sides_separately=True,
            n_workers=n_workers,
            n_thresh=3,
            overwrite=False,
            last_i=None,
            preproc_flag=True,
            thresh_factors=None,
        )

    # 2) Mask QC
    if run_qc:
        mask_qc_wrapper(
            root=root,
            project=project_name,
            mask_type=seg_type,
            n_workers=n_workers,
            overwrite00=False,
            overwrite01=True,
            skip_surf_filtering=False,
        )

    # 3) Geometry fit (sphere trend)
    if run_surface_fit:
        fit_surf_sphere_trend(
            root=root,
            project_name=project_name,
            seg_type=seg_type,
            n_workers=n_workers,
            overwrite=True,
        )

    # 4) Tracking
    if run_tracking:
        perform_tracking(
            root=root,
            project_name=project_name,
            seg_type=seg_type,
            tracking_config=tracking_config,
            start_i=0,
            stop_i=None,
            par_seg_flag=True,
            overwrite_tracking=True,
            use_optical_flow=use_optical_flow,
            well_num=well_num,
        )

    # 5) Fluorescence extraction
    if run_fluorescence:
        extract_foreground_intensities(
            root,
            project_name,
            n_workers=n_workers,
            overwrite=False,
            **(foreground_kwargs or {}),
        )
        compute_mean_fluo_from_foreground(
            root=root,
            project_name=project_name,
            tracking_config=tracking_config,
            n_workers=n_workers,
            overwrite=False,
        )

    # 6) Track feature building + classification
    if run_feature_build:
        build_tracked_mask_features(
            root=root,
            project_name=project_name,
            seg_type=seg_type,
            tracking_config=tracking_config,
            well_num=well_num,
            use_foreground=True,
            used_optical_flow=use_optical_flow,
            n_workers=n_workers,
            mask_field=mask_field,
            process_dropped_nuclei=run_dropped_search,
        )

    if run_classification and classifier_path is not None:
        classify_cell_tracks(
            root=root,
            project_name=project_name,
            tracking_config=tracking_config,
            used_optical_flow=use_optical_flow,
            classifier_path=classifier_path,
            classify_dropped_nuclei=run_dropped_search,
        )

    # 7) Optional smoothing + dropped nuclei pass
    if run_track_smoothing:
        smooth_tracks_wrapper(
            root=root,
            project_name=project_name,
            tracking_config=tracking_config,
            used_flow=use_optical_flow,
            n_workers=n_workers,
            overwrite=False,
            tracking_range=None,
        )

    if run_dropped_search:
        find_dropped_nuclei(
            root=root,
            project_name=project_name,
            tracking_config=tracking_config,
            used_optical_flow=use_optical_flow,
            overwrite=False,
            n_workers=n_workers,
        )

    # 8) Cluster tracking for symmetry-breaking analyses
    if run_cluster_tracking:
        tracks_df, sphere_df = _load_track_data(
            root=root,
            project_name=project_name,
            tracking_config=tracking_config,
        )

        tracks_df = tracks_df.loc[tracks_df["track_class"] == 0].copy()
        tracks_df = tracks_df[["t", "track_id", "x", "y", "z", "mean_fluo"]]

        cluster_config = ClusterTrackingConfig(d_thresh=25.0)
        clusters_by_t = find_clusters_per_timepoint(
            tracks_df,
            sphere_df,
            config=cluster_config,
            fluo_col="mean_fluo",
            time_col="t",
            sphere_radius_col="radius_smooth",
            sphere_center_cols=("center_z_smooth", "center_y_smooth", "center_x_smooth"),
        )

        cluster_ts, _merges_df = track_clusters_over_time(
            clusters_by_t,
            config=cluster_config,
        )

        stitched_ts, stitch_log = stitch_tracklets(
            cluster_ts,
            config=cluster_config,
        )

        out_dir = root / "symmetry_breaking" / project_name / tracking_config
        out_dir.mkdir(parents=True, exist_ok=True)
        stitched_ts.to_csv(out_dir / "cell_clusters_stitched.csv", index=False)
        (out_dir / "cluster_tracking_log.json").write_text(stitch_log.to_json(orient="records"))

        cfg_dict = asdict(cluster_config)
        with open(out_dir / "cluster_tracking_config.yaml", "w") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=False)

