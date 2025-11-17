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
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from collections import defaultdict
from tqdm import tqdm

def preprocess_tracks(df, ROLL_W=5, MOVE_THRESH=1.0,
                      OVERLAP_MIN=5, MERGE_DIST=5.0):

    df = df.sort_values(["track_id", "t"]).copy()

    # ============================================================
    # Track length
    # ============================================================
    df["track_len"] = df.groupby("track_id")["t"].transform("count")

    # ============================================================
    # Rolling step length → stationary frames
    # ============================================================
    df["dx"] = df.groupby("track_id")["x"].diff()
    df["dy"] = df.groupby("track_id")["y"].diff()
    df["dz"] = df.groupby("track_id")["z"].diff()

    df["step_len"] = np.sqrt(df.dx**2 + df.dy**2 + df.dz**2)

    df["roll_move"] = (
        df.groupby("track_id")["step_len"]
          .rolling(ROLL_W, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )

    df["is_stationary"] = df["roll_move"] < MOVE_THRESH
    df["track_mostly_stationary"] = (
        df.groupby("track_id")["is_stationary"].transform(lambda x: x.mean() > 0.7)
    )

    # ============================================================
    # Build time → (track_id, xyz) lookup
    # ============================================================
    # grouped_t = df.groupby("t")

    # For each frame, map track_id → row index
    # t_track = {t: g.track_id.to_numpy() for t, g in grouped_t}
    # t_xyz    = {t: g[["x", "y", "z"]].to_numpy() for t, g in grouped_t}
    #
    # # Storage for spatial overlaps: (tid1, tid2) → count of frames with proximity
    # close_counts = defaultdict(int)

    # ============================================================
    # Spatial overlap detection via cKDTree per frame
    # ============================================================
    # for t in tqdm(sorted(t_xyz.keys()), desc="Flagging overlapping tracks..."):
    #     coords = t_xyz[t]
    #     tids   = t_track[t]
    #
    #     if len(coords) < 2:
    #         continue
    #
    #     tree = cKDTree(coords)
    #     # Query all pairs within MERGE_DIST
    #     idxs = tree.query_ball_tree(tree, MERGE_DIST)
    #
    #     # accumulate per-track proximity hits
    #     for i, neigh in enumerate(idxs):
    #         tid_i = tids[i]
    #         for j in neigh:
    #             if j <= i:        # avoid double counting / self
    #                 continue
    #             tid_j = tids[j]
    #
    #             # count this frame as "paired"
    #             close_counts[(tid_i, tid_j)] += 1
    #
    # # ============================================================
    # # Convert proximity counts to merge edges
    # # ============================================================
    # merge_edges = [
    #     (a, b) for (a, b), cnt in close_counts.items()
    #     if cnt >= OVERLAP_MIN
    # ]
    #
    # # ============================================================
    # # Build adjacency for merging
    # # ============================================================
    # adj = defaultdict(set)
    # for a, b in merge_edges:
    #     adj[a].add(b)
    #     adj[b].add(a)
    #
    # # Connected components
    # visited = set()
    # components = []
    #
    # for tid in df.track_id.unique():
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
    #         stack.extend(adj[u] - visited)
    #     components.append(sorted(comp))
    #
    # # ============================================================
    # # Build fast index lookup (track_id -> row indices)
    # # ============================================================
    # track_index = (
    #     df
    #     .reset_index()
    #     .groupby("track_id")["index"]
    #     .apply(np.array)
    #     .to_dict()
    # )
    #
    # # Pre-extract needed columns once
    # cols = ["t", "x", "y", "z", "step_len", "roll_move",
    #         "parent_track_id", "is_stationary", "track_mostly_stationary"]
    # data = df[cols].to_numpy()
    # col_idx = {c: i for i, c in enumerate(cols)}  # name -> column index
    # t_col = col_idx["t"]
    #
    # # ============================================================
    # # Fuse tracks
    # # ============================================================
    # fused_rows = []
    #
    # for comp in tqdm(components, desc="Fusing tracks..."):
    #     if len(comp) == 1:
    #         # simple case: keep as-is
    #         idx = track_index[comp[0]]
    #         subdf = df.iloc[idx]
    #         fused_rows.append(subdf)
    #         continue
    #
    #     # collect ALL row indices from tracks in the component
    #     all_idx = np.concatenate([track_index[tid] for tid in comp])
    #
    #     sub = data[all_idx]  # ndarray of all rows
    #     sub_t = sub[:, t_col]
    #
    #     # group by t WITHOUT pandas
    #     # -------------------------------------------------------
    #     # 1. sort by time
    #     order = np.argsort(sub_t)
    #     sub = sub[order]
    #     sub_t = sub[:, t_col]
    #
    #     # 2. find frame boundaries
    #     uniq_t, start = np.unique(sub_t, return_index=True)
    #     # next array for fast slicing
    #     end = np.r_[start[1:], len(sub)]
    #
    #     # 3. aggregate manually (fast, vectorized)
    #     # -------------------------------------------------------
    #     out = []
    #     for s, e in zip(start, end):
    #         block = sub[s:e]
    #         out.append([
    #             uniq_t[s],
    #             block[:, col_idx["x"]].mean(),
    #             block[:, col_idx["y"]].mean(),
    #             block[:, col_idx["z"]].mean(),
    #             block[:, col_idx["step_len"]].mean(),
    #             block[:, col_idx["roll_move"]].mean(),
    #             block[:, col_idx["parent_track_id"]].max(),
    #             block[:, col_idx["is_stationary"]].all(),
    #             block[:, col_idx["track_mostly_stationary"]].all(),
    #         ])
    #
    #     out = pd.DataFrame(
    #         out,
    #         columns=[
    #             "t", "x", "y", "z", "step_len", "roll_move",
    #             "parent_track_id", "is_stationary", "track_mostly_stationary"
    #         ]
    #     )
    #     out["track_id"] = comp[0]
    #     fused_rows.append(out)
    #
    # clean_df = pd.concat(fused_rows, ignore_index=True)
    clean_df = df.sort_values(["track_id", "t"]).reset_index(drop=True)

    # clean_df["track_len"] = clean_df.groupby("track_id")["t"].transform("count")
    return clean_df



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
