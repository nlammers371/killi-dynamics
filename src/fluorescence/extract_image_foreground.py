import numpy as np
import zarr
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from src.data_io.zarr_utils import open_experiment_array, open_mask_array
# ===============================================================
# 🧩 Worker helper — runs per timepoint
# ===============================================================
def _extract_foreground_single(
    t: int,
    root: Path,
    project_name: str,
    seg_type: str,
    mask_field: str,
    use_gpu: bool,
    well_num: int | None,
    mask_store_path: Path,
    side_spec: str,
    fg_group_name: str,
):
    """Worker: extract sparse foreground coordinates and intensities for a single timepoint."""
    # Reopen zarrs inside each worker for process isolation
    mask_zarr, _, _ = open_mask_array(
        root=root,
        project_name=project_name,
        side="fused",
        seg_type=seg_type,
        mask_field=mask_field,
        well_num=well_num,
        use_gpu=use_gpu,
    )
    img_zarr, _, _ = open_experiment_array(
        root=root, project_name=project_name, well_num=well_num, use_gpu=use_gpu
    )

    mask_frame = mask_zarr[t]
    if mask_frame.max() == 0:
        return

    im_frame = img_zarr[t]
    fg_coords = np.argwhere(mask_frame > 0)

    if im_frame.ndim == 4:  # (C, Z, Y, X)
        C = im_frame.shape[0]
        values = np.stack([im_frame[c][fg_coords[:, 0], fg_coords[:, 1], fg_coords[:, 2]] for c in range(C)], axis=1)
    else:
        values = im_frame[fg_coords[:, 0], fg_coords[:, 1], fg_coords[:, 2]][..., None]

    # Each worker reopens its output group
    fg_root = zarr.open_group(mask_store_path / side_spec / fg_group_name, mode="a")
    g = fg_root.create_group(f"t{t:04d}")
    g.create_dataset(
        "coords",
        data=fg_coords.astype(np.int32),
        chunks=(min(len(fg_coords), 100_000), 3),
    )
    g.create_dataset(
        "values",
        data=values.astype(np.float32),
        chunks=(min(len(fg_coords), 100_000), values.shape[1]),
    )


# ===============================================================
# Main driver — sets up the structure and runs in parallel
# ===============================================================
def extract_foreground_intensities(
    root: Path,
    project_name: str,
    seg_type: str = "li_segmentation",
    mask_field: str = "clean",
    overwrite: bool = False,
    well_num: int | None = None,
    n_workers: int = 1,
):
    """
    Extract sparse foreground voxel coordinates and intensity values for each timepoint,
    and store them as a hierarchical group inside the existing segmentation Zarr.
    """

    root = Path(root)

    # --- open segmentation zarr group ---
    mask_zarr, mask_store_path, _ = open_mask_array(
        root=root,
        project_name=project_name,
        side="fused",
        seg_type=seg_type,
        mask_field=mask_field,
        well_num=well_num,
    )

    side_spec = "fused"
    n_t = mask_zarr.shape[0]
    group = zarr.open_group(mask_store_path / side_spec, mode="a")
    scale_vec = group.attrs["voxel_size_um"]

    # --- open image zarr ---
    img_zarr, _, _ = open_experiment_array(
        root=root, project_name=project_name, well_num=well_num
    )  # will look for fused by default
    channels = img_zarr.attrs["channels"]
    n_channels = len(channels)

    # --- set up output group path ---
    fg_group_name = f"foreground_{mask_field}"
    if fg_group_name in group:
        if overwrite:
            del group[fg_group_name]
        else:
            print(f"[extract_foreground_intensities] Using existing group '{fg_group_name}'")
            return mask_store_path / side_spec

    fg_root = group.create_group(fg_group_name)
    fg_root.attrs.update({
        "description": "Sparse per-frame foreground voxel coordinates and intensities",
        "n_channels": n_channels,
        "voxel_size_um": scale_vec,
    })

    call_extract_foreground = partial(_extract_foreground_single,
                                        root=root,
                                        project_name=project_name,
                                        seg_type=seg_type,
                                        mask_field=mask_field,
                                        well_num=well_num,
                                        mask_store_path=mask_store_path,
                                        side_spec=side_spec,
                                        fg_group_name=fg_group_name,
                                        use_gpu= n_workers == 1)

    # --- run parallel or serial ---
    if n_workers > 1:
        process_map(call_extract_foreground,
                    range(n_t),
                    max_workers=n_workers,
                    chunksize=1,
                    desc="Extracting foreground intensities...")
    else:
        for t in tqdm(range(n_t), desc="Extracting foreground intensities..."):
            call_extract_foreground(t)

    print(f"[extract_foreground_intensities] Done — stored under {mask_store_path}/{side_spec}/{fg_group_name}")
    return mask_store_path
