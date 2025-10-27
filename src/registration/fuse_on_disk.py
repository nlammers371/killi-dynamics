import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import shift as ndi_shift
from tqdm import tqdm
from pathlib import Path
from warnings import warn

def fuse_single_frame_all_channels(
    t,
    img_ref,
    img_mov,
    shifts,
    mov_flip_axes=(False, False, False),
    blend=True,
):
    """Fuse all channels for one frame according to precomputed shifts."""
    multichannel = len(img_ref.shape) > 4  # TCZYX

    if multichannel:
        n_channels = img_ref.shape[1]
        fused_channels = []
        for c in range(n_channels):
            ref_frame = np.squeeze(img_ref[t, c])
            mov_frame = np.squeeze(img_mov[t, c])

            # flip moving side
            for ax, flip in enumerate(mov_flip_axes):
                if flip:
                    mov_frame = np.flip(mov_frame, axis=ax)

            # apply shift
            shift_vec = shifts[t]
            mov_shifted = ndi_shift(mov_frame, shift_vec, order=1, mode="nearest")

            # combine
            if blend:
                fused_frame = np.mean([ref_frame, mov_shifted], axis=0)
            else:
                fused_frame = np.maximum(ref_frame, mov_shifted)

            fused_channels.append(fused_frame.astype(ref_frame.dtype))

        fused_stack = np.stack(fused_channels, axis=0)
    else:
        # single-channel case (TZYX)
        ref_frame = np.squeeze(img_ref[t])
        mov_frame = np.squeeze(img_mov[t])

        for ax, flip in enumerate(mov_flip_axes):
            if flip:
                mov_frame = np.flip(mov_frame, axis=ax)

        shift_vec = shifts[t]
        mov_shifted = ndi_shift(mov_frame, shift_vec, order=1, mode="nearest")

        if blend:
            fused_stack = np.mean([ref_frame, mov_shifted], axis=0)
        else:
            fused_stack = np.maximum(ref_frame, mov_shifted)

        fused_stack = fused_stack.astype(ref_frame.dtype)

    return fused_stack


def fuse_hemisperes_on_disk(
        zarr_root: Path,
        ref_side="side_00",
        mov_side="side_01",
        overwrite=False,
        mov_flip_axes=(True, False, True),
        blend=True):

    """
    Physically fuse all channels of two sides using per-frame rigid shifts.
    Output written to <dataset>.zarr/fused/.
    """
    warn("This fusion script still needs to be tested...")
    zarr_root = Path(zarr_root)

    # --- load source arrays ---
    img_ref = zarr.open(zarr_root / ref_side, mode="r")
    img_mov = zarr.open(zarr_root / mov_side, mode="r")
    multichannel = len(img_ref.shape) > 4

    # --- load shifts from registration inside same store ---
    reg_csv = zarr_root / "registration" / f"shifts_{mov_side}.csv"
    shifts = pd.read_csv(reg_csv)[["zs", "ys", "xs"]].to_numpy()

    n_frames = min(img_ref.shape[0], img_mov.shape[0])

    # --- prepare fused group ---
    root_group = zarr.open_group(zarr_root, mode="a")
    if "fused" in root_group:
        if overwrite:
            print("⚠️ Removing existing fused group before rewrite.")
            del root_group["fused"]
        else:
            raise RuntimeError("Fused group already exists. Use overwrite=True to replace.")
    fused_group = root_group.create_group("fused")

    # infer output shape
    fused_shape = img_ref.shape
    chunks = img_ref.chunks if hasattr(img_ref, "chunks") else None

    fused = fused_group.zeros(
        "data",
        shape=fused_shape,
        dtype=img_ref.dtype,
        chunks=chunks,
    )

    # --- perform fusion ---
    for t in tqdm(range(n_frames), desc="Fusing frames"):
        fused_frame = fuse_single_frame_all_channels(
            t,
            img_ref,
            img_mov,
            shifts,
            mov_flip_axes=mov_flip_axes,
            blend=blend,
        )
        if multichannel:
            fused[t, :, :, :, :] = fused_frame
        else:
            fused[t, :, :, :] = fused_frame

    # --- metadata ---
    fused_group.attrs.update({
        "source_sides": [ref_side, mov_side],
        "dim_order": "TCZYX" if multichannel else "TZYX",
        "registration_source": str(reg_csv.relative_to(zarr_root)),
        "mov_flip_axes": mov_flip_axes,
        "blend_mode": "mean" if blend else "max"
    })

    print(f"✅ All-channel fusion written to {zarr_root}/fused/")
    return fused
