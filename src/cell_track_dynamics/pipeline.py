from __future__ import annotations
import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))
"""Top-level orchestration for the :mod:`cell_field_dynamics` analysis pipeline."""

from pathlib import Path
from typing import Any
from src.data_io.zarr_io import get_metadata
from src.data_io.track_io import _load_track_data, _load_tracks
from src.cell_track_dynamics.density import compute_surface_density
from src.cell_track_dynamics.velocity_corr import compute_local_velocity_alignment, compute_windowed_velocity_corr
from src.tracking.track_processing import add_sphere_coords_to_tracks
from src.tracking.track_processing import preprocess_tracks

def run(
    root: Path,
    project_name: str,
    track_config_name: str,
    flows_flag: bool = True,
    n_workers: int = 1,
    deep_cells_only: bool = True,
    fluo_col: str | None = None,
    ball_radius: float = 50.0,
    remove_stationary: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute the lagrangian cell-dynamics analysis pipeline.

    The current implementation focuses on setting up the computation graph and
    file layout described in the project brief. Numerical routines are
    intentionally lightweight placeholders so the module can be exercised in
    downstream notebooks while the full scientific algorithms are developed.
    """

    _, tracking_dir = _load_tracks(root=root,
                                   project_name=project_name,
                                   tracking_config=track_config_name,
                                   prefer_flow=flows_flag)

    tracks, sphere_df = _load_track_data(root=root,
                                            project_name=project_name,
                                            tracking_config=track_config_name,
                                            prefer_flow=flows_flag,
                                            prefer_smoothed=True,)

    if deep_cells_only:
        tracks = tracks[tracks["track_class"] == 0]
        tracks = tracks[(tracks["t"] > 1400) & (tracks["t"] <= 1500)]
        tracks = tracks.reset_index(drop=True)

    # Preprocess tracks
    print("Preprocessing tracks...")
    tracks = preprocess_tracks(tracks)

    if remove_stationary:
        tracks = tracks[~tracks["track_mostly_stationary"]].reset_index(drop=True)

    # sphere_df, tracks = _normalise_tracking_output(data)
    # metadata = get_metadata(root, project_name)
    # tracks["time_min"] = tracks["t"] * metadata["time_resolution_s"] / 60

    # add sphere coordinates to tracks
    # tracks = add_sphere_coords_to_tracks(tracks, sphere_df)

    # initialize metric DF
    metrics = tracks.loc[:, ["track_id", "t"]].copy().reset_index(drop=True)

    # calculate density
    print("Calculating surface densities...")
    impute_list = [fluo_col] if fluo_col is not None else None
    density_nn, imputed_vals = compute_surface_density(df=tracks,
                                                       sphere_df=sphere_df,
                                                       radius=ball_radius,
                                                       var_list=impute_list,
                                                       n_workers=n_workers)
    metrics["density_nn"] = density_nn
    if imputed_vals is not None and fluo_col is not None:
        metrics[fluo_col + "_nn"] = imputed_vals[fluo_col]

    # calculate frame-over frame velocities
    print("Calculating velocity alignment...")
    tracks = tracks.sort_values(by=["track_id", "t"]).reset_index(drop=True)
    tracks[["vx", "vy", "vz"]] = tracks[["x", "y", "z"]].groupby(tracks["track_id"]).diff()
    vel_alignment = compute_local_velocity_alignment(df=tracks,
                                                     radius=ball_radius,
                                                     n_workers=n_workers)
    metrics["vel_alignment"] = vel_alignment

    # windowed velocity correlation
    vcorr = compute_windowed_velocity_corr(tracks,
                                           window=11,
                                           n_workers=n_workers,
                                           radius=ball_radius)

    # write to file
    print("Writing metrics to file...")
    metrics = metrics.sort_values(by=["track_id", "t"]).reset_index(drop=True)
    metrics_file = tracking_dir / "cell_dynamics_metrics.csv"
    metrics.to_csv(metrics_file, index=False)



    # summary["tracks_preview"] = augmented_tracks.head().to_dict(orient="list")

    return metrics


__all__ = ["run"]

if __name__ == "__main__":
    import pprint

    # root = Path(r"Y:\killi_dynamics")
    root = Path("/media/nick/hdd011/killi_dynamics/")
    project_name = "20251019_BC1-NLS_52-80hpf"
    seg_type = "li_segmentation"
    tracking_config = "tracking_20251102"
    flows_flag = True

    result_summary = run(
        root=root,
        project_name=project_name,
        track_config_name=tracking_config,
        flows_flag=flows_flag,
        deep_cells_only=True,
        fluo_col="mean_fluo",
        ball_radius=75.0,
        n_workers=1,
    )
    pprint.pprint(result_summary)
