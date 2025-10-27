import numpy as np
import pandas as pd
import zarr
from skimage.registration import phase_cross_correlation
from tqdm.contrib.concurrent import process_map
from functools import partial
from pathlib import Path
import multiprocessing

def align_halves(t, image_data1, image_data2, z_align_size=50, nucleus_channel=1):
    """Compute per-frame rigid shift between hemispheres."""
    multichannel = len(image_data2.shape) > 4
    if multichannel:
        data_zyx1 = np.squeeze(image_data1[t, nucleus_channel])
        data_zyx2 = np.squeeze(image_data2[t, nucleus_channel])
    else:
        data_zyx1 = np.squeeze(image_data1[t])
        data_zyx2 = np.squeeze(image_data2[t])

    # apply manual flips (mirror across Z and X for opposing halves)
    data_zyx2_i = data_zyx2[::-1, :, ::-1]

    # restrict alignment volume
    align1 = data_zyx1[:z_align_size, :, :]
    align2 = data_zyx2_i[-z_align_size:, :, :]

    shift, error, _ = phase_cross_correlation(
        align1, align2,
        normalization=None,
        upsample_factor=2,
        overlap_ratio=0.05,
    )

    shift_corrected = shift.copy()
    shift_corrected[0] += z_align_size
    return shift_corrected


def get_hemisphere_shifts(zarr_root: Path, ref_side: str = "side_00", mov_side: str = "side_01",
                          interval: int = 25, nucleus_channel: int = 1, z_align_size: int = 50,
                          start_i: int = 0, last_i: int | None = None, n_workers: int | None = None):
    """Estimate per-frame shifts and update Zarr metadata."""
    zarr_root = Path(zarr_root)
    ref_path = zarr_root / ref_side
    mov_path = zarr_root / mov_side
    reg_dir = zarr_root / "registration"
    reg_dir.mkdir(parents=True, exist_ok=True)

    image_data1 = zarr.open(ref_path, mode="r")
    image_data2 = zarr.open(mov_path, mode="r")

    n_frames = min(image_data1.shape[0], image_data2.shape[0])
    if last_i is None:
        last_i = n_frames - 1

    frame_vec = np.arange(start_i, last_i + 1)
    frames_to_register = np.unique(np.arange(start_i, last_i, interval).tolist() + [last_i])

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() // 4)

    align_fun = partial(
        align_halves,
        image_data1=image_data1,
        image_data2=image_data2,
        z_align_size=z_align_size,
        nucleus_channel=nucleus_channel,
    )

    shift_vec = process_map(align_fun, frames_to_register, max_workers=n_workers, chunksize=1)
    shift_array = np.asarray(shift_vec)

    # Interpolate to all frames
    shift_interp = np.empty((len(frame_vec), 3))
    for i in range(3):
        shift_interp[:, i] = np.interp(frame_vec, frames_to_register, shift_array[:, i])

    shift_df = pd.DataFrame({"frame": frame_vec, "zs": shift_interp[:, 0],
                             "ys": shift_interp[:, 1], "xs": shift_interp[:, 2]})

    # ---- SAVE SHIFTS ----
    csv_path = reg_dir / f"shifts_{mov_side}.csv"
    shift_df.to_csv(csv_path, index=False)

    # ---- SAVE SHIFTS ----
    csv_path = reg_dir / f"shifts_{mov_side}.csv"
    shift_df.to_csv(csv_path, index=False)

    # ---- UPDATE MOVING SIDE ATTRS ----
    mov_group = zarr.open_group(mov_path, mode="r+")
    mov_attrs = dict(mov_group.attrs.asdict())
    mov_attrs.setdefault("dim_order", "TCZYX")
    mov_attrs["rigid_transform"] = {
        "flip": [True, False, True],  # match your mirroring scheme
        "per_frame_shifts": str(csv_path.relative_to(zarr_root))
    }
    mov_group.attrs.update(mov_attrs)

    # ---- UPDATE REFERENCE SIDE ATTRS ----
    ref_group = zarr.open_group(ref_path, mode="r+")
    ref_attrs = dict(ref_group.attrs.asdict())
    ref_attrs.setdefault("dim_order", "TCZYX")
    ref_attrs["rigid_transform"] = {
        "flip": [False, False, False],
        "per_frame_shifts": None
    }
    ref_group.attrs.update(ref_attrs)

    print(f"✅ Saved shifts to {csv_path}")
    print(f"✅ Updated {mov_side}/.zattrs with rigid_transform info")


if __name__ == "__main__":
    dataset_path = Path(r"E:\Nick\symmetry_breaking\built_data\zarr_image_files\20251010\example_dataset.zarr")
    get_hemisphere_shifts(dataset_path)
