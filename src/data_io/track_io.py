from pathlib import Path
import pandas as pd
import numpy as np
from src.data_io.zarr_io import open_experiment_array
import re
import warnings

def parse_track_paths(root: Path,
                      project_name: str,
                      tracking_config: str,
                      tracking_range: tuple[int, int] | None = None,
                      prefer_flow: bool = True,):

    tracking_root = root / "tracking" / project_name / tracking_config
    track_stub = "track*"
    if tracking_range is not None:
        track_stub = tracking_root / f"{tracking_range[0]:04d}_{tracking_range[1]:04d}"

    # tracking_results = tracking_results_flow
    tracking_results = list(np.unique(list(tracking_root.glob(track_stub))))
    tracking_results = [d for d in tracking_results if d.is_dir()]
    nf_paths = [p for p in tracking_results if "withflow" not in p.name]
    f_paths = [p for p in tracking_results if "withflow" in p.name]
    if prefer_flow and len(f_paths) > 0:
        tracking_dir = f_paths[0]
        return tracking_dir
    elif prefer_flow and len(nf_paths) > 0:
        warnings.warn(f"Preferred optical flow tracking, but none found in {tracking_root}. Using non-flow tracking.")
    elif not prefer_flow and len(nf_paths) > 0:
        tracking_dir = nf_paths[0]
        return tracking_dir
    elif not prefer_flow and len(f_paths) > 0:
        warnings.warn(f"Preferred non-optical flow tracking, but none found in {tracking_root}. Using flow tracking.")
        tracking_dir = f_paths[0]
        return tracking_dir
    else:
        raise ValueError(f"No tracking results found matching preference in {tracking_root}")




def parse_member_ids(x):
    """Convert malformed string lists like '[1 2 5]' → [1,2,5]."""
    if isinstance(x, list):
        return x
    if not isinstance(x, str) or not x.strip():
        return np.nan
    s = x.strip().strip('[]')
    if not s:
        return []
    # replace commas or multiple spaces with single space
    s = re.sub(r'[,]+', ' ', s)
    parts = re.split(r'\s+', s.strip())
    try:
        return [int(p) for p in parts if p]
    except ValueError:
        # fallback if something weird slipped in
        return [p for p in parts if p]

def _load_tracks(root: Path,
                 project_name: str,
                 tracking_config: str,
                 tracking_range: tuple[int, int] | None = None,
                 prefer_flow: bool = True,
                 prefer_smoothed: bool = True
                 ):

    # --- open segmentation and experiment arrays ---
    tracking_dir = parse_track_paths(root=root,
                                      project_name=project_name,
                                      tracking_config=tracking_config,
                                      tracking_range=tracking_range,
                                      prefer_flow=prefer_flow)

    if prefer_smoothed and (tracking_dir / "tracks_smoothed.csv").is_file():
        tracks = pd.read_csv(tracking_dir / "tracks_smoothed.csv")
    else:
        tracks = pd.read_csv(tracking_dir / "tracks.csv")

    return tracks, tracking_dir


def _load_track_data(root: Path,
                     project_name: str,
                     tracking_config: str,
                     tracking_range: tuple[int, int] | None = None,
                     prefer_flow: bool = True,
                     prefer_smoothed: bool = True,
                     rescale_tracks: bool = True,
                     fluo_channel_to_use: int | None = None,
                     add_dropped_nuclei: bool = True
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
    tracks, tracking_dir = _load_tracks(root=root,
                                        project_name=project_name,
                                        tracking_config=tracking_config,
                                        tracking_range=tracking_range,
                                        prefer_smoothed=prefer_smoothed,
                                        prefer_flow=prefer_flow)

    if add_dropped_nuclei:
        dropped_file = tracking_dir / "dropped_nuclei.csv"
        if dropped_file.is_file():
            dropped_df = pd.read_csv(dropped_file).drop(columns=["mask_id_src"], errors="ignore")
            dropped_df["dropped_cell"] = True
            tracks["dropped_cell"] = False
            tracks = pd.concat([tracks, dropped_df], ignore_index=True, axis=0)
    scale_vec = np.asarray(image_store.attrs["voxel_size_um"])
    if rescale_tracks:
        tracks[["z", "y", "x"]] = tracks[["z", "y", "x"]].multiply(scale_vec[None, :], axis=1)

    # load track class info (if present)
    class_file = tracking_dir / "track_class_df.csv"
    if class_file.is_file():
        class_df = pd.read_csv(class_file)
        tracks = tracks.merge(class_df, on="track_id", how="left")
    dropped_class_file = tracking_dir / "dropped_nuclei_class_df.csv"
    if dropped_class_file.is_file():
        dropped_class_df = pd.read_csv(dropped_class_file)
        tracks = tracks.merge(dropped_class_df, on=["track_id", "t"], how="left", suffixes=("", "_dropped"))


    # load fluo data
    fluo_file = tracking_dir / "tracks_fluo.csv"
    if fluo_file.is_file():
        fluo_df = pd.read_csv(fluo_file)
        fluo_df = fluo_df[fluo_df["channel"] == fluo_channel_to_use]
        tracks = tracks.merge(fluo_df, on=["track_id", "t"], how="left")

    # check for cluster tracking data
    cluster_path = root / "symmetry_breaking" / project_name / tracking_config / "cell_clusters_stitched.csv"
    if cluster_path.is_file():
        cluster_df = pd.read_csv(cluster_path)

        cluster_df["member_track_id"] = cluster_df["member_track_id"].apply(parse_member_ids)

        cluster_assignments = (
            cluster_df[["t", "cluster_id_stitched", "member_track_id"]]
            .explode("member_track_id")
            .rename(columns={"member_track_id": "track_id"})
        )

        # ensure numeric dtype if possible
        cluster_assignments["track_id"] = pd.to_numeric(cluster_assignments["track_id"], errors="coerce").astype(
            "Int64")

        tracks = tracks.merge(cluster_assignments, on=["t", "track_id"], how="left")

    # sphere data
    sphere_path = root / "surf_stats" / f"{project_name}_surf_stats.zarr" / "surf_fits" / "sphere_fits.csv"
    if sphere_path.is_file():
        sphere_df = pd.read_csv(sphere_path)
    else:
        sphere_df = pd.DataFrame()

    return tracks, sphere_df