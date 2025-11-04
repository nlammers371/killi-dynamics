from pathlib import Path
import numpy as np
import dask.array as da
import zarr
from ultrack import load_config, track, to_tracks_layer, tracks_to_zarr
import shutil
import warnings

from src.tracking.tracking_utils import labels_to_contours_nl
from src.data_io.zarr_utils import open_mask_array
# --- assumed imports from your codebase ---
# from your_module import load_config, labels_to_contours_nl, track, to_tracks_layer, tracks_to_zarr


def perform_tracking(
    root: Path,
    project_name: str,
    tracking_config: str,
    seg_type: str,
    overwrite_tracking: bool = False,
    overwrite_segmentation: bool = False,
    well_num: int | None = None,
    start_i: int = 0,
    stop_i: int | None = None,
    use_stack_flag: bool = False,
    use_marker_masks: bool = False,
    suffix: str = "",
    par_seg_flag: bool = True,
):
    """
    Execute Ultrack on a mask time series.
    """
    # --- setup ---

    # --- mask path selection ---
    # mask_dir = root / "segmentation" / seg_type
    # mask_path = mask_dir / f"{file_prefix}_masks.zarr"
    # mask_store = zarr.open(mask_path, mode="r")

    # load mask data
    mask_field = "thresh_stacks" if use_stack_flag else "clean"
    mask_zarr, mask_store_path, side_spec = open_mask_array(root=root,
                                          project_name=project_name,
                                          seg_type=seg_type,
                                          mask_field=mask_field,
                                          well_num=well_num)

    mask_da = da.from_zarr(mask_zarr)

    if stop_i is None:
        stop_i = mask_da.shape[0]

    if use_marker_masks:
        project_name += "_marker"

    # --- define output directories ---
    seg_path = mask_store_path / side_spec
    project_path = root / "tracking" / project_name / tracking_config
    if well_num is not None:
        project_path = project_path / f"well{well_num:04}"

    project_sub_path = project_path / f"track_{start_i:04}_{stop_i:04}{suffix}"
    project_sub_path.mkdir(parents=True, exist_ok=True)

    if "voxel_size_um" in mask_zarr.attrs:
        scale_vec: list[float] = mask_zarr.attrs["voxel_size_um"]
    else:  # for backwards compatibility
        scale_vec = [mask_zarr.attrs[k] for k in ("PhysicalSizeZ", "PhysicalSizeY", "PhysicalSizeX")]

    # check to see if tracing data exists
    # Delete if it exists
    if project_sub_path.exists() | overwrite_tracking:
        shutil.rmtree(project_sub_path)
    else:
        warnings.warn(f"Output directory {project_sub_path} already exists. Set overwrite=True to replace.")
        return

    # Recreate empty directory
    project_sub_path.mkdir(parents=True, exist_ok=True)

    # --- load configuration ---
    metadata_path = root / "metadata" / "tracking"
    cfg = load_config(metadata_path / f"{tracking_config}.txt")
    cfg.data_config.working_dir = str(project_sub_path)

    # copy the file
    src = root / "metadata" / "tracking" / f"{tracking_config}.txt"
    dst = project_path / f"{tracking_config}.txt"
    shutil.copy2(src, dst)

    # --- initialize segmentation stores ---
    seg_store = zarr.open_group(seg_path, mode="a")

    # --- create or open arrays directly at the root ---
    if "detection" not in seg_store or overwrite_segmentation:
        if "detection" in seg_store:
            del seg_store["detection"]
        detection = seg_store.create_dataset(
            "detection",
            shape=mask_da.shape,
            dtype=bool,
            chunks=(1,) + mask_da.shape[1:],
            overwrite=True,
        )
    else:
        detection = seg_store["detection"]

    if "boundaries" not in seg_store or overwrite_segmentation:
        if "boundaries" in seg_store:
            del seg_store["boundaries"]
        boundaries = seg_store.create_dataset(
            "boundaries",
            shape=mask_da.shape,
            dtype=np.uint16,
            chunks=(1,) + mask_da.shape[1:],
            overwrite=True,
        )
    else:
        boundaries = seg_store["boundaries"]

    # --- find missing frames to process ---
    segment_indices = np.arange(start_i, stop_i)
    dstore_path = seg_path / "detection"
    if not overwrite_segmentation:
        existing_files = list(dstore_path.iterdir()) if dstore_path.exists() else []
        written = {int(p.name.split(".")[0]) for p in existing_files if p.name.split(".")[0].isdigit()}
        segment_indices = np.array(sorted(set(segment_indices) - written))

    # --- segmentation and boundary extraction ---
    if segment_indices.size > 0:
        detection_da, boundaries_da = labels_to_contours_nl(
            mask_da,
            segment_indices,
            par_flag=par_seg_flag,
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
    tracks_csv_path = project_sub_path / "tracks.csv"
    tracks_df.to_csv(tracks_csv_path, index=False)

    segments_path = project_sub_path / "segments.zarr"
    segments = tracks_to_zarr(cfg, tracks_df, store_or_path=str(segments_path), overwrite=True)

    print("Done.")
    return segments
