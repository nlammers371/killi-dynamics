from pathlib import Path
import pandas as pd
import numpy as np
from src.data_io.zarr_io import open_experiment_array

def _load_tracks(root: Path,
                 project_name: str,
                 tracking_config: str,
                 tracking_range: tuple[int, int] | None = None,
                 prefer_smoothed: bool = True
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

    if prefer_smoothed and (tracking_dir / "tracks_smoothed.csv").is_file():
        tracks = pd.read_csv(tracking_dir / "tracks_smoothed.csv")
    else:
        tracks = pd.read_csv(tracking_dir / "tracks.csv")

    return tracks, tracking_dir


def _load_track_data(root: Path,
                     project_name: str,
                     tracking_config: str,
                     tracking_range: tuple[int, int] | None = None,
                     prefer_smoothed: bool = True,
                     rescale_tracks: bool = True,
                     fluo_channel_to_use: int | None = None
                 ):

    # load image store (for metadata)
    image_store, _, _ = open_experiment_array(
        root=root,
        project_name=project_name,
        well_num=None
    )

    if fluo_channel_to_use is None:
        channel_list = image_store.attrs["channels"]
        nuclear_channel = next(
            i for i, ch in enumerate(channel_list)
            if ("H2B" in ch.upper()) or ("NLS" in ch.upper())
        )
        fluo_channel_to_use = [i for i in range(len(channel_list)) if i != nuclear_channel][0]

    # load tracks
    tracks, tracking_dir = _load_tracks(root, project_name, tracking_config, tracking_range, prefer_smoothed)

    scale_vec = np.asarray(image_store.attrs["voxel_size_um"])
    if rescale_tracks:
        tracks[["z", "y", "x"]] = tracks[["z", "y", "x"]].multiply(scale_vec[None, :], axis=1)

    # load track class info (if present)
    class_file = tracking_dir / "track_class_df.csv"
    if class_file.is_file():
        class_df = pd.read_csv(class_file)
        tracks = tracks.merge(class_df, on="track_id", how="left")

    # load fluo data
    fluo_file = tracking_dir / "tracks_fluo.csv"
    if fluo_file.is_file():
        fluo_df = pd.read_csv(fluo_file)
        fluo_df = fluo_df[fluo_df["channel"] == fluo_channel_to_use]
        tracks = tracks.merge(fluo_df, on=["track_id", "t"], how="left")

    # sphere data
    sphere_path = root / "surf_stats" / f"{project_name}_surf_stats.zarr" / "surf_fits" / "sphere_fits.csv"
    if sphere_path.is_file():
        sphere_df = pd.read_csv(sphere_path)
    else:
        sphere_df = pd.DataFrame()

    return tracks, sphere_df