from tqdm.contrib.concurrent import process_map
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path
from src.data_io.zarr_io import open_experiment_array
from src.data_io.track_io import _load_tracks
from functools import partial


def _smooth_single_track(group: pd.DataFrame,
                         coord_cols: list[str],
                         sg_window_frames: int,
                         sg_poly: int) -> pd.DataFrame:
    """Apply Savitzky–Golay smoothing to a single track group and return a copy."""
    n = len(group)
    if n < 3:
        return group

    window = min(sg_window_frames, n)
    if window % 2 == 0:
        window = max(3, window - 1)
    if window < 3:
        return group

    group = group.copy()
    for col in coord_cols:
        group[col] = savgol_filter(
            group[col].to_numpy(float),
            window_length=window,
            polyorder=min(sg_poly, window - 1),
            mode="interp",
        )
    return group


def smooth_tracks(tracks: pd.DataFrame,
                  dT: float,
                  sg_window_minutes: float = 5,
                  sg_poly: int = 2,
                  n_workers: int = 8) -> pd.DataFrame:
    """Apply Savitzky–Golay smoothing to Cartesian coordinates per track in parallel."""

    if tracks.empty:
        return tracks.copy()

    coord_cols = [c for c in ("x", "y", "z") if c in tracks.columns]
    if len(coord_cols) != 3:
        return tracks.copy()

    if "track_id" not in tracks or not any(c in tracks for c in ("time_min", "t")):
        raise ValueError("Requires 'track_id' and time column ('time_min' or 't').")

    time_col = "time_min" if "time_min" in tracks else "t"

    # prepare parameters
    tracks = tracks.sort_values(["track_id", time_col])
    sg_window_frames = int(sg_window_minutes / dT)
    if sg_window_frames % 2 == 0:
        sg_window_frames += 1

    # group per track
    groups = [g for _, g in tracks.groupby("track_id")]
    # Define the partial once (picklable)
    worker_fn = partial(_smooth_single_track,
                        coord_cols=coord_cols,
                        sg_window_frames=sg_window_frames,
                        sg_poly=sg_poly)
    # parallel smoothing
    if n_workers > 1:
        smoothed_groups = process_map(
            worker_fn,
            groups,
            max_workers=n_workers,
            chunksize=1,
            desc="Smoothing tracks (parallel)",
            unit="track",
        )
    else:
        smoothed_groups = [
            _smooth_single_track(g, coord_cols, sg_window_frames, sg_poly) for g in groups
        ]

    smoothed = pd.concat(smoothed_groups, ignore_index=True)

    return smoothed

def smooth_tracks_wrapper(  root: Path,
                            project_name: str,
                            tracking_config: str,
                            tracking_range: tuple[int, int] | None = None,
                            n_workers: int = 1,
                            sg_window_minutes: float = 5,
                            sg_poly: int = 2,
                            ) -> pd.DataFrame:

    tracks, tracking_dir = _load_tracks(root, project_name, tracking_config, tracking_range)
    image_store, _, _ = open_experiment_array(root=root, project_name=project_name, well_num=None, use_gpu=False)
    tres_min = image_store.attrs["time_resolution_s"] / 60
    smoothed_tracks = smooth_tracks(tracks,
                                    dT=tres_min,
                                    n_workers=n_workers,
                                    sg_window_minutes=sg_window_minutes,
                                    sg_poly=sg_poly)

    smoothed_tracks = smoothed_tracks.loc[:, ["track_id", "parent_track_id", "t", "x", "y", "z"]]

    smoothed_tracks.to_csv(tracking_dir / "tracks_smooth.csv", index=False)

    return smoothed_tracks



def add_sphere_coords_to_tracks(tracks_df: pd.DataFrame, sphere_df: pd.DataFrame) -> pd.DataFrame:

    smoothed_centers = sphere_df.loc[:, ["t", "center_x_smooth", "center_y_smooth", "center_z_smooth", "radius_smooth"]]
    tracks_df = tracks_df.merge(smoothed_centers, on="t", how="left")
    rel = tracks_df[["x", "y", "z"]].to_numpy() - tracks_df[
        ["center_x_smooth", "center_y_smooth", "center_z_smooth"]].to_numpy()
    r = np.sqrt(np.einsum("ij,ij->i", rel, rel))
    theta = np.arccos(np.clip(rel[:, 2] / r, -1, 1))
    phi = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * np.pi)
    tracks_df["r"] = r
    tracks_df["theta"] = theta
    tracks_df["phi"] = phi

    return tracks_df