from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from src.tracking.track_processing import _load_tracks

def _mean_fluo_foreground_single(
    t: int,
    mask_zarr_path: Path,
    fg_group_path: Path,
) -> pd.DataFrame:

    """Compute per-mask, per-channel mean fluorescence for one frame."""
    mask_zarr = zarr.open(mask_zarr_path, mode="r")
    mask = mask_zarr[t]
    if mask.max() == 0:
        return pd.DataFrame(columns=["t", "mask_id", "channel", "mean_fluo"])

    g_t = zarr.open_group(fg_group_path / f"t{t:04d}", mode="r")
    coords = g_t["coords"][:]            # (N, 3)
    values = g_t["values"][:]            # (N, C)
    labels = mask[coords[:, 0], coords[:, 1], coords[:, 2]]

    valid = labels > 0
    coords, values, labels = coords[valid], values[valid], labels[valid]

    df_list = []
    for c in range(values.shape[1]):
        vals = values[:, c]
        df_c = (
            pd.DataFrame({"mask_id": labels, "val": vals})
            .groupby("mask_id", as_index=False)["val"]
            .mean()
            .rename(columns={"val": "mean_fluo"})
        )
        df_c["t"] = t
        df_c["channel"] = c
        df_list.append(df_c)

    return pd.concat(df_list, ignore_index=True)


def compute_mean_fluo_from_foreground(
    root: Path,
    project_name: str,
    tracking_config: str,
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
    fg_group_path = mask_zarr_path_src.parent / f"foreground_{mask_field}"

    if not fg_group_path.exists():
        raise FileNotFoundError(f"No foreground data found at {fg_group_path}")

    mask_store = zarr.open(mask_zarr_path_src, mode="r")
    n_t = mask_store.shape[0]
    # load tracks
    if tracking_config is not None:
        _, tracking_dir = _load_tracks(root, project_name, tracking_config, tracking_range)
        mask_zarr_path_dst = tracking_dir / "segments.zarr"
    else:
        mask_zarr_path_dst = mask_zarr_path_src

    n_channels = zarr.open_group(fg_group_path / "t0000", mode="r")["values"].shape[1]

    run_frame = partial(
        _mean_fluo_foreground_single,
        mask_zarr_path=mask_zarr_path_dst,
        fg_group_path=fg_group_path,
    )

    if n_workers > 1:
        dfs = process_map(run_frame, range(n_t),
                          max_workers=n_workers, chunksize=1,
                          desc="Computing mean fluorescence...")
    else:
        dfs = [run_frame(t) for t in tqdm(range(n_t), desc="Computing mean fluorescence...")]

    out_df = pd.concat([df for df in dfs if not df.empty], ignore_index=True)
    out_df = out_df.rename(columns={"mask_id": "track_id"})
    # tracks = tracks.merge(out_df, on=["track_id", "t"], how="left")
    if tracking_config is not None:
        out_df.to_csv(tracking_dir / "tracks_fluo.csv", index=False)
    else:
        print("[compute_mean_fluo_from_foreground] No tracking config given, not saving output.")
        # pass

    return out_df
