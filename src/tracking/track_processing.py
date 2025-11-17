from tqdm.contrib.concurrent import process_map
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path
from src.data_io.zarr_io import open_experiment_array
from src.data_io.track_io import _load_tracks
from functools import partial
import zarr
from skimage.measure import regionprops_table


def preprocess_tracks(df, ROLL_W=5, MOVE_THRESH=1.0, OVERLAP_MIN=5, MERGE_DIST=5.0):
    df = df.sort_values(["track_id", "t"]).copy()

    # ============================================================
    # (ii) Add track length (frames)
    # ============================================================
    df["track_len"] = df.groupby("track_id")["t"].transform("count")

    # ============================================================
    # (i) Flag stationary frames using rolling mean step length
    # ============================================================
    # frame-to-frame displacement per track
    df["dx"] = df.groupby("track_id")["x"].diff()
    df["dy"] = df.groupby("track_id")["y"].diff()
    df["dz"] = df.groupby("track_id")["z"].diff()
    df["step_len"] = np.sqrt(df["dx"]**2 + df["dy"]**2 + df["dz"]**2)

    # rolling mean step length (ignore NaNs)
    df["roll_move"] = (
        df.groupby("track_id")["step_len"]
          .rolling(ROLL_W, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )

    df["is_stationary"] = df["roll_move"] < MOVE_THRESH

    # track-level summary (optional)
    df["track_mostly_stationary"] = (
        df.groupby("track_id")["is_stationary"]
          .transform(lambda x: x.mean() > 0.7)
    )

    # ============================================================
    # (iii) Fuse duplicate / parallel tracks
    # ============================================================
    # Identify overlapping + spatially co-located tracks
    # track_ids = df["track_id"].unique()
    # track_bounds = (
    #     df.groupby("track_id")["t"]
    #     .agg(["min", "max"])
    #     .rename(columns={"min": "t0", "max": "t1"})
    # )

    # # build adjacency list for track-merge graph
    # merge_edges = []
    #
    # for i, tid1 in enumerate(track_ids):
    #     t0_1, t1_1 = track_bounds.loc[tid1, ["t0", "t1"]]
    #     df1 = df[df.track_id == tid1]
    #
    #     for tid2 in track_ids[i+1:]:
    #         t0_2, t1_2 = track_bounds.loc[tid2, ["t0", "t1"]]
    #
    #         # temporal overlap
    #         overlap_start = max(t0_1, t0_2)
    #         overlap_end   = min(t1_1, t1_2)
    #         if overlap_end - overlap_start < OVERLAP_MIN:
    #             continue
    #
    #         # spatial proximity check
    #         df2 = df[df.track_id == tid2]
    #         merged = df1.merge(df2, on="t", suffixes=("_1","_2"))
    #         if len(merged) < OVERLAP_MIN:
    #             continue
    #
    #         dist = np.sqrt(
    #             (merged["x_1"]-merged["x_2"])**2
    #             + (merged["y_1"]-merged["y_2"])**2
    #             + (merged["z_1"]-merged["z_2"])**2
    #         )
    #
    #         if dist.median() < MERGE_DIST:
    #             merge_edges.append((tid1, tid2))
    #
    # # build connected components of merge graph
    # # (each component = tracks that belong to one cell)
    # adj = {}
    # for a, b in merge_edges:
    #     adj.setdefault(a, set()).add(b)
    #     adj.setdefault(b, set()).add(a)
    #
    # # DFS to get connected components
    # visited = set()
    # components = []
    #
    # for tid in track_ids:
    #     if tid in visited:
    #         continue
    #     stack = [tid]
    #     comp = []
    #     while stack:
    #         u = stack.pop()
    #         if u in visited:
    #             continue
    #         visited.add(u)
    #         comp.append(u)
    #         for v in adj.get(u, []):
    #             if v not in visited:
    #                 stack.append(v)
    #     components.append(sorted(comp))
    #
    # # ============================================================
    # # Fuse tracks inside each component
    # # ============================================================
    # fused_rows = []
    #
    # for comp in components:
    #     if len(comp) == 0:
    #         continue
    #     elif len(comp) == 1:
    #         fused_rows.append(df[df.track_id == comp[0]])
    #         continue
    #
    #     # choose earliest track ID as representative
    #     rep = comp[0]
    #
    #     # pull all rows for all tracks in this component
    #     sub = df[df.track_id.isin(comp)]
    #
    #     # fuse by averaging positions per frame
    #     fused = (
    #         sub.groupby("t")
    #            .agg({"x": "mean", "y": "mean", "z": "mean",
    #                  "step_len": "mean",
    #                  "roll_move": "mean",
    #                  "parent_track_id": "max",  # keep highest parent ID
    #                  "is_stationary": "all",
    #                  "track_mostly_stationary": "all"})  # any = true if any frame stationary
    #            .reset_index()
    #     )
    #     fused["track_id"] = rep
    #     fused_rows.append(fused)
    #
    # clean_df = pd.concat(fused_rows, ignore_index=True)
    # clean_df = clean_df.sort_values(["track_id", "t"]).reset_index(drop=True)
    #
    # # recompute track_len after fusion
    # clean_df["track_len"] = clean_df.groupby("track_id")["t"].transform("count")

    # clean_df["track_id"] = clean_df["track_id"].astype(int)
    # clean_df.loc[np.isnan(clean_df["parent_track_id"]), "parent_track_id"] = -1
    # clean_df["parent_track_id"] = clean_df["parent_track_id"].astype(int)

    return df
def _smooth_single_track(group: pd.DataFrame,
                         coord_cols: list[str],
                         sg_window_frames: int,
                         sg_poly: int):
    """Apply Savitzky–Golay smoothing to a single track group and return a copy."""
    n = len(group)
    if n < 3:
        return group.to_dict("records")

    window = min(sg_window_frames, n)
    if window % 2 == 0:
        window = max(3, window - 1)
    if window < 3:
        return group.to_dict("records")

    group = group.copy()
    for col in coord_cols:
        group[col] = savgol_filter(
            group[col].to_numpy(float),
            window_length=window,
            polyorder=min(sg_poly, window - 1),
            mode="interp",
        )
    return group.to_dict("records")


def smooth_tracks(tracks: pd.DataFrame,
                  dT: float,
                  sg_window_minutes: float = 5,
                  sg_poly: int = 2,
                  n_workers: int = 8):

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

    # flatten list-of-lists → list-of-dicts-
    rows = [row for chunk in smoothed_groups for row in chunk]
    smoothed = pd.DataFrame(rows)

    return smoothed

def smooth_tracks_wrapper(  root: Path,
                            project_name: str,
                            tracking_config: str,
                            tracking_range: tuple[int, int] | None = None,
                            used_flow: bool = True,
                            n_workers: int = 1,
                            sg_window_minutes: float = 5,
                            sg_poly: int = 2,
                            overwrite: bool = False,
                            ) -> pd.DataFrame:

    tracks, tracking_dir = _load_tracks(root, project_name, tracking_config, tracking_range, prefer_flow=used_flow)
    image_store, _, _ = open_experiment_array(root=root, project_name=project_name, well_num=None, use_gpu=False)
    tres_min = image_store.attrs["time_resolution_s"] / 60

    out_path = tracking_dir / "tracks_smoothed.csv"
    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)

    smoothed_tracks = smooth_tracks(tracks,
                                    dT=tres_min,
                                    n_workers=n_workers,
                                    sg_window_minutes=sg_window_minutes,
                                    sg_poly=sg_poly)

    smoothed_tracks = smoothed_tracks.loc[:, ["track_id", "parent_track_id", "t", "x", "y", "z"]]

    smoothed_tracks.to_csv(out_path, index=False)

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

def _compare_masks_single_frame(t,
                                 mask_zarr_path_src: Path,
                                 mask_zarr_path_tracks: Path,) -> pd.DataFrame:
    """Compare masks between source and tracked segments for a single frame."""
    mask_store_src = zarr.open(mask_zarr_path_src, mode="r")
    mask_store_tracks = zarr.open(mask_zarr_path_tracks, mode="r")
    mask_src = mask_store_src[t]
    mask_tracks = mask_store_tracks[t]
    if mask_src.max() == 0:
        return pd.DataFrame(columns=["t", "mask_id_src", "mask_id_track", "overlap_volume_um3"])

    props = regionprops_table(mask_src, properties=("label", "coords", "centroid"))
    records = []
    n = len(props["label"])
    for i in range(n):
        label = props["label"][i]
        coords = props["coords"][i]  # (N,3)
        centroid = np.array([
            props["centroid-0"][i],
            props["centroid-1"][i],
            props["centroid-2"][i],
        ])

        # get corresponding labels in tracked mask
        tracked_labels, counts = np.unique(mask_tracks[tuple(coords.T)], return_counts=True)
        foreground_counts = counts[tracked_labels != 0]
        if np.sum(foreground_counts) / np.sum(counts) < 0.1:

            records.append({
                "t": t,
                "mask_id_src": label,
                "z": centroid[0],
                "y": centroid[1],
                "x": centroid[2],
            })

    return pd.DataFrame(records)

def find_dropped_nuclei(root: Path,
    project_name: str,
    tracking_config: str,
    used_optical_flow: bool = True,
    tracking_range: tuple[int, int] | None = None,
    seg_type: str = "li_segmentation",
    side_key: str = "fused",
    mask_field: str = "clean",
    n_workers: int = 1,
    overwrite: bool = False,

) -> pd.DataFrame:
    """
    Compute per-mask, per-channel mean fluorescence using sparse foreground arrays.

    Reads pre-extracted `foreground_<mask_field>` groups and corresponding mask zarrs.
    """

    mask_zarr_path_src = (
        root / "segmentation" / seg_type / f"{project_name}_masks.zarr" / side_key / mask_field
    )

    mask_store = zarr.open(mask_zarr_path_src, mode="r")
    n_t = mask_store.shape[0]
    # load tracks
    tracks, tracking_dir = _load_tracks(root, project_name, tracking_config, tracking_range, prefer_flow=used_optical_flow)
    mask_zarr_path_tracks = tracking_dir / "segments.zarr"

    # define write dir
    out_path = tracking_dir / "dropped_nuclei.csv"
    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)

    # look for masks in original that are not in segments
    run_mask_check = partial(_compare_masks_single_frame,
                            mask_zarr_path_src=mask_zarr_path_src,
                            mask_zarr_path_tracks=mask_zarr_path_tracks)

    if n_workers > 1:
        df_list = process_map(
            run_mask_check,
            range(n_t),
            max_workers=n_workers,
            chunksize=1,
            desc="Finding dropped nuclei (parallel)",
            unit="frame",
        )
    else:
        df_list = [run_mask_check(t) for t in range(n_t)]

    # save
    dropped_nuclei_df = pd.concat(df_list, ignore_index=True)
    start_id = np.max(tracks["track_id"]) + 1
    dropped_nuclei_df["track_id"] = np.arange(start_id, start_id + len(dropped_nuclei_df))
    dropped_nuclei_df.to_csv(out_path, index=False)

    return dropped_nuclei_df
