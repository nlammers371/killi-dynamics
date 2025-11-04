"""Top-level orchestration for the :mod:`cell_field_dynamics` analysis pipeline."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
from src.cell_field_dynamics.cd_utils import add_sphere_coords_to_tracks
from src.data_io.zarr_io import get_metadata
from src.cell_field_dynamics.config import (
    GridConfig,
    MaterialsConfig,
    NoiseConfig,
    QCConfig,
    RunPaths,
    SmoothingConfig,
    WindowConfig,
)
from src.cell_field_dynamics import flux, grids, io_functions, materials, metrics, msd, qc, vector_field
from src.cell_field_dynamics.cd_utils import load_tracking_data


# def _normalise_tracking_output(data: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """Normalise the return signature from :func:`load_tracking_data`."""
#
#     if isinstance(data, tuple):
#         if len(data) == 2:
#             first, second = data
#         elif len(data) >= 3:
#             first, second = data[:2]
#         else:
#             raise ValueError("Unexpected tuple length returned by load_tracking_data")
#     else:
#         raise TypeError("load_tracking_data must return a tuple of DataFrames")
#
#     if "particle" in getattr(first, "columns", []):
#         tracks_df, sphere_df = first, second
#     else:
#         sphere_df, tracks_df = first, second
#
#     if not isinstance(tracks_df, pd.DataFrame) or not isinstance(sphere_df, pd.DataFrame):
#         raise TypeError("load_tracking_data must return pandas.DataFrame objects")

    # return sphere_df, tracks_df


def run(
    root: Path,
    project_name: str,
    track_config_name: str,
    grid_cfg: GridConfig = GridConfig(),
    win_cfg: WindowConfig = WindowConfig(),
    smooth_cfg: SmoothingConfig = SmoothingConfig(),
    qc_cfg: QCConfig = QCConfig(),
    noise_cfg: NoiseConfig = NoiseConfig(),
    mat_cfg: MaterialsConfig = MaterialsConfig(),
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute the cell-dynamics analysis pipeline.

    The current implementation focuses on setting up the computation graph and
    file layout described in the project brief. Numerical routines are
    intentionally lightweight placeholders so the module can be exercised in
    downstream notebooks while the full scientific algorithms are developed.
    """

    out_root = root / "cell_field_dynamics" / project_name / track_config_name
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    load_kwargs = {
        "root": root,
        "project_name": project_name,
        "track_config_name": track_config_name,
    }

    tracks_df, sphere_df, class_df = load_tracking_data(**load_kwargs)

    # sphere_df, tracks_df = _normalise_tracking_output(data)
    metadata = get_metadata(root, project_name)
    smoothed_tracks = vector_field.smooth_tracks(tracks_df, smooth_cfg, metadata["time_resolution_s"] / 60)
    smoothed_tracks = add_sphere_coords_to_tracks(smoothed_tracks, sphere_df)
    # smoothed_tracks.drop(columns=["center_x_smooth", "center_y_smooth", "center_z_smooth", "radius_smooth"], inplace=True)

    grid_indexers = grids.build_healpix_indexers(grid_cfg.nsides)
    binned_tracks = grids.bin_tracks_over_time(
        smoothed_tracks,
        grid_indexers,
        grid_cfg,
        win_cfg,
    )

    step_table = vector_field.build_step_table(smoothed_tracks)

    vector_results = vector_field.compute_vector_field(
        smoothed_tracks, binned_tracks, win_cfg, smooth_cfg, step_table=step_table
    )
    metric_results = metrics.compute_scalar_metrics(
        smoothed_tracks, vector_results, binned_tracks, win_cfg, step_table=step_table
    )
    msd_results = msd.compute_msd_metrics(smoothed_tracks, binned_tracks, win_cfg, step_table=step_table)
    material_results = materials.compute_material_metrics(
        smoothed_tracks, binned_tracks, mat_cfg, win_cfg, step_table=step_table
    )
    flux_results = flux.compute_flux_from_vector_field(
        vector_results, radius=step_table.mean_radius
    )

    qc_results = qc.apply_quality_control(
        binned_tracks,
        metric_results,
        vector_results,
        msd_results,
        material_results,
        flux_results,
        qc_cfg,
    )

    zarr_paths = io_functions.write_zarr_stores(
        out_root,
        grid_indexers,
        binned_tracks,
        vector_results,
        metric_results,
        msd_results,
        material_results,
        flux_results,
        qc_results,
        grid_cfg,
        win_cfg,
        smooth_cfg,
        noise_cfg,
    )

    augmented_tracks, tracks_path = io_functions.augment_tracks_df(
        smoothed_tracks,
        grid_indexers,
        metric_results,
        msd_results,
        material_results,
        flux_results,
        qc_results,
        grid_cfg,
        out_root,
        overwrite=overwrite,
    )

    summary: dict[str, Any] = {
        "paths": RunPaths(out_root=out_root, zarr_paths=zarr_paths, tracks_table=tracks_path),
        "grid_cfg": asdict(grid_cfg),
        "window_cfg": asdict(win_cfg),
        "smoothing_cfg": asdict(smooth_cfg),
        "qc_cfg": asdict(qc_cfg),
        "noise_cfg": asdict(noise_cfg),
        "materials_cfg": asdict(mat_cfg),
        "counts": {
            nside: result.counts.astype(int).sum(axis=0).tolist()
            for nside, result in binned_tracks.items()
        },
    }

    summary["tracks_preview"] = augmented_tracks.head().to_dict(orient="list")

    return summary


__all__ = ["run"]

if __name__ == "__main__":
    import pprint

    root = Path(r"E:\pipeline_dev\killi_dynamics")
    project_name = "MEM_NLS_test"
    tracking_config = "tracking_v0"
    result_summary = run(
        root=root,
        project_name=project_name,
        track_config_name=tracking_config,
        grid_cfg=GridConfig(nsides=[8, 16]),
        win_cfg=WindowConfig(win_minutes=30.0, stride_minutes=10.0),
    )
    pprint.pprint(result_summary)
