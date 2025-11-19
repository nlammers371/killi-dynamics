"""Top-level orchestration for the :mod:`cell_field_dynamics` analysis pipeline."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
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
from src.cell_field_dynamics import qc, vector_field, metrics, flux, density, grids, io_functions
from src.data_io.track_io import _load_track_data
from src.tracking.track_processing import add_sphere_coords_to_tracks, smooth_tracks


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
    flows_flag: bool = True,
    n_workers: int = 1,
    deep_cells_only: bool = True,
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

    tracks_df, sphere_df = _load_track_data(root=root,
                                            project_name=project_name,
                                            tracking_config=track_config_name,
                                            prefer_flow=flows_flag,)
    if deep_cells_only:
        tracks_df = tracks_df[tracks_df["track_class"] == 0]
        tracks_df = tracks_df[(tracks_df["t"] > 1250) & (tracks_df["t"] <= 1300)]
        # tracks_df = tracks_df[(tracks_df["t"] <= 5)]
        tracks_df = tracks_df.reset_index(drop=True)

    # sphere_df, tracks_df = _normalise_tracking_output(data)
    metadata = get_metadata(root, project_name)
    tracks_df["time_min"] = tracks_df["t"] * metadata["time_resolution_s"] / 60
    smoothed_tracks = smooth_tracks(tracks_df,
                                    dT=metadata["time_resolution_s"] / 60,
                                    n_workers=n_workers,
                                    sg_window_minutes=smooth_cfg.sg_window_minutes,
                                    sg_poly=smooth_cfg.sg_poly)

    # smoothed_tracks = smooth_tracks(tracks_df, smooth_cfg, metadata["time_resolution_s"] / 60)
    smoothed_tracks = add_sphere_coords_to_tracks(smoothed_tracks, sphere_df)
    step_table = vector_field.build_step_table(smoothed_tracks, fluo_col="mean_fluo")
    sphere_radius = step_table.mean_radius

    # update smoothing config with sphere radius
    smooth_cfg.sphere_radius_um = sphere_radius

    grid_indexers, neighbor_indexers = grids.build_healpix_indexers(grid_cfg.nsides,
                                                                    smooth_cfg=smooth_cfg)
    binned_tracks = grids.bin_tracks_over_time(
                                            smoothed_tracks,
                                            grid_indexers,
                                            grid_cfg,
                                            win_cfg,
                                        )

    density_results = density.compute_healpix_density_field(
        step_table,
        binned_tracks,
        win_cfg,
        neighbors=neighbor_indexers,
        smooth_cfg=smooth_cfg,
        n_workers=n_workers,
    )

    vector_results = vector_field.compute_vector_field(
                                                    smoothed_tracks,
                                                    binned_tracks,
                                                    win_cfg,
                                                    smooth_cfg=smooth_cfg,
                                                    neighbors=neighbor_indexers,
                                                    step_table=step_table,
                                                    n_workers=n_workers
                                                )

    metric_results = metrics.compute_scalar_metrics(
                                    smoothed_tracks,
                                    vector_results,
                                    binned_tracks,
                                    win_cfg,
                                    smooth_cfg=smooth_cfg,
                                    step_table=step_table,
                                    neighbors=neighbor_indexers,
                                )
    # msd_results = msd.compute_msd_metrics(smoothed_tracks, binned_tracks, win_cfg, step_table=step_table)
    # material_results = materials.compute_material_metrics(
    #     smoothed_tracks, binned_tracks, mat_cfg, win_cfg, step_table=step_table
    # )
    flux_results = flux.compute_flux_from_vector_field(
        vector_results, radius=step_table.mean_radius
    )

    # qc_results = qc.apply_quality_control(
    #     grid_
    #     # metric_results,
    #     # vector_results,
    #     # density_results,
    #     # msd_results,
    #     # material_results,
    #     flux_results,
    #     qc_cfg,
    # )

    zarr_paths = io_functions.write_zarr_stores(
        out_root=out_root,
        indexers=grid_indexers,
        grid_results=binned_tracks,
        flux_results=flux_results,
        vector_results=vector_results,
        metric_results=metric_results,
        density_results=density_results,
        grid_cfg=grid_cfg,
        win_cfg=win_cfg,
        smooth_cfg=smooth_cfg,
        noise_cfg=noise_cfg,
    )

    # augmented_tracks, tracks_path = io_functions.augment_tracks_df(
    #     smoothed_tracks,
    #     grid_indexers,
    #     metric_results,
    #     # msd_results,
    #     # material_results,
    #     flux_results,
    #     qc_results,
    #     grid_cfg,
    #     out_root,
    #     overwrite=overwrite,
    # )

    # summary: dict[str, Any] = {
    #     "paths": RunPaths(out_root=out_root, zarr_paths=zarr_paths, tracks_table=tracks_path),
    #     "grid_cfg": asdict(grid_cfg),
    #     "window_cfg": asdict(win_cfg),
    #     "smoothing_cfg": asdict(smooth_cfg),
    #     "qc_cfg": asdict(qc_cfg),
    #     "noise_cfg": asdict(noise_cfg),
    #     "materials_cfg": asdict(mat_cfg),
    #     "counts": {
    #         nside: result.counts.astype(int).sum(axis=0).tolist()
    #         for nside, result in binned_tracks.items()
    #     },
    # }

    # summary["tracks_preview"] = augmented_tracks.head().to_dict(orient="list")

    return {}


__all__ = ["run"]

if __name__ == "__main__":
    import pprint

    root = Path(r"Y:\killi_dynamics")
    project_name = "20251019_BC1-NLS_52-80hpf"
    seg_type = "li_segmentation"
    tracking_config = "tracking_20251102"
    flows_flag = True

    result_summary = run(
        root=root,
        project_name=project_name,
        track_config_name=tracking_config,
        flows_flag=flows_flag,
        grid_cfg=GridConfig(nsides=[8]),
        smooth_cfg=SmoothingConfig(sigma_space_um=45.0),
        win_cfg=WindowConfig(win_minutes=9.0, stride_minutes=3),
        n_workers=1,
    )
    pprint.pprint(result_summary)
