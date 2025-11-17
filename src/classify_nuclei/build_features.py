import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from skimage.measure import regionprops
import zarr
from src.data_io.zarr_io import open_mask_array, open_experiment_array
from src.tracking.track_processing import _load_tracks
from functools import partial
import warnings

# -------------------------------------------------------------------------
# --- Helper: per-region feature extraction (same geometry logic as before)
# -------------------------------------------------------------------------
def extract_region_features(region) -> dict:
    """Compute geometric and intensity features for a single region."""
    vol = region.area
    convex_vol = getattr(region, "convex_area", np.nan)
    solidity = vol / convex_vol if convex_vol and convex_vol > 0 else np.nan
    extent = getattr(region, "extent", np.nan)

    mean_intensity = getattr(region, "mean_intensity", np.nan)
    min_intensity = getattr(region, "min_intensity", np.nan)
    max_intensity = getattr(region, "max_intensity", np.nan)

    eigvals = getattr(region, "inertia_tensor_eigvals", None)
    if eigvals is not None and len(eigvals) == 3 and eigvals[0] != 0:
        elongation_ratio = eigvals[-1] / eigvals[0]
    else:
        elongation_ratio = np.nan

    if hasattr(region, "bbox") and len(region.bbox) == 6:
        min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
        bbox_z, bbox_y, bbox_x = max_z - min_z, max_y - min_y, max_x - min_x
    else:
        bbox_z = bbox_y = bbox_x = np.nan

    return dict(
        label=region.label,
        volume=vol,
        convex_volume=convex_vol,
        solidity=solidity,
        extent=extent,
        mean_intensity=mean_intensity,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
        elongation_ratio=elongation_ratio,
        bbox_z=bbox_z,
        bbox_y=bbox_y,
        bbox_x=bbox_x,
    )

# -------------------------------------------------------------------------
# --- Helper: process a single frame
# -------------------------------------------------------------------------
def process_frame(
    t: int,
    seg_zarr,
    img_zarr=None,
    tracks_df=None,
    label_field="mask_id_src",
    scale_vec=None,
    nuclear_channel=1,
    use_foreground=False,
    fg_group=None,
) -> pd.DataFrame:
    """Compute per-region features for one frame, optionally using sparse foreground data."""

    seg = np.asarray(seg_zarr[t]).squeeze()

    # --------------------------
    # 0) Track-based region mask
    # --------------------------
    if tracks_df is not None:
        valid_labels = set(
            tracks_df.loc[tracks_df["t"] == t, label_field].astype(int).tolist()
        )
    else:
        valid_labels = None

    # =====================================================================
    # --- 1) STANDARD CASE (full intensity image)
    # =====================================================================
    if not use_foreground:
        if img_zarr is None:
            raise ValueError("img_zarr is required when not using foreground mode.")

        img = np.asarray(img_zarr[t, nuclear_channel]).squeeze()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*convex hull.*")
            regions = regionprops(seg, intensity_image=img, spacing=scale_vec)

        # --- FILTER REGIONS HERE ---
        if valid_labels is not None:
            regions = [r for r in regions if r.label in valid_labels]

        df = pd.DataFrame([extract_region_features(r) for r in regions])
        df["frame"] = t
        return df

    # =====================================================================
    # --- 2) SPARSE FOREGROUND CASE
    # =====================================================================
    fg_t_key = f"t{t:04d}"
    if fg_t_key not in fg_group:
        return pd.DataFrame(columns=["label", "frame"])

    # --- 2A: geometry (regionprops without image) ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*convex hull.*")
        regions_geom = regionprops(seg, spacing=scale_vec)

    # --- FILTER REGIONS HERE ---
    if valid_labels is not None:
        regions_geom = [r for r in regions_geom if r.label in valid_labels]

    # No regions left? Return empty
    if len(regions_geom) == 0:
        return pd.DataFrame(columns=[
            "label","volume","convex_volume","solidity","extent",
            "mean_intensity","min_intensity","max_intensity",
            "elongation_ratio","bbox_z","bbox_y","bbox_x","frame"
        ])

    df_geom = pd.DataFrame([extract_region_features(r) for r in regions_geom])

    # --- 2B: intensity (sparse coords/values) ---
    coords = np.asarray(fg_group[fg_t_key]["coords"])
    values = np.asarray(fg_group[fg_t_key]["values"])[:, nuclear_channel]

    labels = seg[tuple(coords.T)]
    mask = labels > 0
    labels = labels[mask]
    values = values[mask]

    if labels.size == 0:
        df_geom["mean_intensity"] = np.nan
        df_geom["min_intensity"] = np.nan
        df_geom["max_intensity"] = np.nan
        df_geom["frame"] = t
        return df_geom

    df_intensity = (
        pd.DataFrame({"label": labels, "val": values})
        .groupby("label", as_index=False)
        .agg(mean_intensity=("val", "mean"),
             min_intensity=("val", "min"),
             max_intensity=("val", "max"))
    )

    # --- 2C: merge + finalize ---
    df = pd.merge(df_geom, df_intensity, on="label", how="left")
    df["frame"] = t

    ordered_cols = [
        "label",
        "volume", "convex_volume", "solidity", "extent",
        "mean_intensity", "min_intensity", "max_intensity",
        "elongation_ratio", "bbox_z", "bbox_y", "bbox_x",
        "frame",
    ]
    return df.reindex(columns=ordered_cols)



def build_tracked_mask_features(
        root: Path | str,
        project_name: str,
        tracking_config: str | None,
        tracking_range: tuple[int, int] | None = None,
        used_optical_flow: bool = True,
        nuclear_channel: int = None,
        seg_type: str = "li_segmentation",
        mask_field: str = "clean",
        n_workers: int = 1,
        use_foreground: bool = False,
        well_num: int | None = None,
        process_dropped_nuclei: bool = False,
) -> pd.DataFrame:
    """
    Build per-mask feature table, optionally using sparse foreground zarrs for intensity features.
    """
    root = Path(root)

    # --- open segmentation and experiment arrays ---
    _, tracking_dir = _load_tracks(
                        root,
                        project_name,
                        tracking_config,
                        tracking_range,
                        prefer_flow=used_optical_flow,
                    )
    if not process_dropped_nuclei:
        seg_zarr = zarr.open(tracking_dir / "segments.zarr", mode="r")
    else:
        seg_zarr_path_src = (
            root / "segmentation" / seg_type / f"{project_name}_masks.zarr" / "fused" / mask_field
        )
        seg_zarr = zarr.open(seg_zarr_path_src, mode="r")

    n_t = seg_zarr.shape[0]
    stop_i = tracking_range[1] if tracking_range is not None else n_t
    start_i = tracking_range[0] if tracking_range is not None else 0
    frames = range(start_i, stop_i)

    # open image or foreground source
    if use_foreground:
        _, seg_store_path, side_key = open_mask_array(root=root,
                                                      project_name=project_name,
                                                      seg_type=seg_type,
                                                      side="fused")
        fg_group_name = f"foreground_{mask_field}"
        seg_store = zarr.open_group(seg_store_path / "fused", mode="r")
        if fg_group_name not in seg_store:
            raise FileNotFoundError(
                f"No foreground group '{fg_group_name}' found in {seg_store_path}/{side_key}."
            )
        fg_group = seg_store[fg_group_name]
        img_zarr = None
        meta = seg_store.attrs
    else:
        img_zarr, _, _ = open_experiment_array(root=root, project_name=project_name, well_num=well_num)
        fg_group = None
        meta = img_zarr.attrs

    scale_vec = np.array(seg_store.attrs["voxel_size_um"])
    if nuclear_channel is None:
        channel_list = meta.get("channels")
        nuclear_channel = next(
            i for i, ch in enumerate(channel_list)
            if ("H2B" in ch.upper()) or ("NLS" in ch.upper())
        )
        # raise ValueError("nuclear_channel must be specified.")

    if process_dropped_nuclei:
        tracks_df = pd.read_csv(tracking_dir / "dropped_nuclei.csv")
    else:
        tracks_df = None

    # --- run per-frame ---
    worker_fn = partial(process_frame,
                        seg_zarr=seg_zarr,
                        img_zarr=img_zarr,
                        tracks_df=tracks_df,
                        scale_vec=scale_vec,
                        nuclear_channel=nuclear_channel,
                        use_foreground=use_foreground,
                        fg_group=fg_group,
                    )

    if n_workers > 1:
        dfs = process_map(worker_fn, frames, max_workers=n_workers, chunksize=1, desc="Extracting mask features")
    else:
        dfs = [worker_fn(t) for t in tqdm(frames, desc="Extracting mask features")]

    # --- combine ---
    feature_df = pd.concat(dfs, ignore_index=True)
    if not process_dropped_nuclei:
        feature_df.to_csv(tracking_dir / "mask_features.csv", index=False)
    else:
        feature_df.to_csv(tracking_dir / "dropped_nuclei_features.csv", index=False)

    return feature_df
