from pathlib import Path
import numpy as np
import dask.array as da
import zarr
from track_utils import labels_to_contours_nl
from ultrack import load_config, track, to_tracks_layer, tracks_to_zarr

# --- assumed imports from your codebase ---
# from your_module import load_config, labels_to_contours_nl, track, to_tracks_layer, tracks_to_zarr


def perform_tracking(
    root: Path,
    project_name: str,
    tracking_config: str,
    seg_model: str,
    well_num: int | None = None,
    start_i: int = 0,
    stop_i: int | None = None,
    use_stack_flag: bool = False,
    use_marker_masks: bool = False,
    use_fused: bool = True,
    suffix: str = "",
    par_seg_flag: bool = True,
    last_filter_start_i: int | None = None,
):
    """
    Execute Ultrack on a mask time series.
    """
    # --- setup ---
    tracking_name: str = tracking_config.replace(".txt", "")

    if well_num is not None:
        file_prefix: str = f"{project_name}_well{well_num:04}"
        subfolder: Path = Path(project_name)
    else:
        file_prefix = project_name
        subfolder = Path()
        well_num = 0

    # --- mask path selection ---
    mask_dir: Path = root / "built_data" / "mask_stacks" / seg_model / subfolder
    if use_stack_flag:
        mask_path: Path = mask_dir / f"{file_prefix}_mask_stacks.zarr"
    elif use_fused:
        mask_path = mask_dir / f"{file_prefix}_mask_fused.zarr"
    elif use_marker_masks:
        mask_path = mask_dir / f"{file_prefix}_marker_masks.zarr"
    else:
        mask_path = mask_dir / f"{file_prefix}_mask_aff.zarr"

    metadata_path: Path = root / "metadata" / "tracking"
    mask_store = zarr.open(mask_path, mode="a")
    mask_da = da.from_zarr(mask_store)

    if stop_i is None:
        stop_i = mask_store.shape[0]

    if use_marker_masks:
        project_name += "_marker"

    # --- define output directories ---
    seg_path: Path = root / "tracking" / project_name / "segmentation" / f"well{well_num:04}"
    project_path: Path = root / "tracking" / project_name / tracking_name / f"well{well_num:04}"
    project_sub_path: Path = project_path / f"track_{start_i:04}_{stop_i:04}{suffix}"
    project_sub_path.mkdir(parents=True, exist_ok=True)

    # --- metadata propagation if needed ---
    if not mask_store.attrs and use_fused:
        ref_path: Path = mask_dir / f"{file_prefix}_side1_mask_aff.zarr"
        ref_store = zarr.open(ref_path, mode="r")
        for key, value in ref_store.attrs.items():
            mask_store.attrs[key] = value

    if "voxel_size_um" in mask_store.attrs:
        scale_vec: list[float] = mask_store.attrs["voxel_size_um"]
    else:
        scale_vec = [mask_store.attrs[k] for k in ("PhysicalSizeZ", "PhysicalSizeY", "PhysicalSizeX")]

    # --- load configuration ---
    cfg = load_config(metadata_path / f"{tracking_config}.txt")
    cfg.data_config.working_dir = str(project_sub_path)

    # --- initialize segmentation stores ---
    dstore_path: Path = seg_path / "detection.zarr"
    bstore_path: Path = seg_path / "boundaries.zarr"

    dstore = zarr.DirectoryStore(str(dstore_path))
    bstore = zarr.DirectoryStore(str(bstore_path))

    detection = zarr.open(
        store=dstore,
        mode="a",
        shape=mask_store.shape,
        dtype=bool,
        chunks=(1,) + mask_store.shape[1:],
    )
    boundaries = zarr.open(
        store=bstore,
        mode="a",
        shape=mask_store.shape,
        dtype=np.uint16,
        chunks=(1,) + mask_store.shape[1:],
    )

    # --- find missing frames to process ---
    segment_indices = np.arange(start_i, stop_i)
    existing_files = list(dstore_path.iterdir()) if dstore_path.exists() else []
    written = {int(p.stem) for p in existing_files if p.stem.isdigit()}
    segment_indices = np.array(sorted(set(segment_indices) - written))

    # --- segmentation and boundary extraction ---
    if segment_indices.size > 0:
        detection_da, boundaries_da = labels_to_contours_nl(
            mask_da,
            segment_indices,
            par_flag=par_seg_flag,
            last_filter_start_i=last_filter_start_i,
            scale_vec=scale_vec,
        )
        for t, frame in enumerate(segment_indices):
            detection[frame] = detection_da[t]
            boundaries[frame] = boundaries_da[t]

    # --- tracking ---
    detection_da = da.from_zarr(detection)[start_i:stop_i]
    boundaries_da = da.from_zarr(boundaries)[start_i:stop_i]

    print("Performing tracking...")
    track(cfg, detection=detection_da, edges=boundaries_da, scale=scale_vec)

    # --- save results ---
    print("Saving results...")
    tracks_df, graph = to_tracks_layer(cfg)
    tracks_csv_path: Path = project_sub_path / "tracks.csv"
    tracks_df.to_csv(tracks_csv_path, index=False)

    segments_path: Path = project_sub_path / "segments.zarr"
    segments = tracks_to_zarr(cfg, tracks_df, store_or_path=str(segments_path), overwrite=True)

    print("Done.")
    return segments
