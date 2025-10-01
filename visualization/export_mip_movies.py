#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations
import os
from pathlib import Path
from typing import Sequence, Literal, Optional

import numpy as np
import zarr
import dask.array as da
import imageio  # v2 API (still maintained!)
import imageio.v3 as iio
# pip install imageio[ffmpeg]
from tqdm import tqdm
from typing import Mapping, Sequence

# default:   ch-0 → magenta,  ch-1 → green
SWAP_MAP = {
    0: (0, 1, 0),   # channel-0 → green
    1: (1, 0, 1),   # channel-1 → magenta
}

def _to_rgb(
    frame_ctyx: np.ndarray,
    *,
    channel_map: Mapping[int, Sequence[float]] = SWAP_MAP,
    clip: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Convert (C,Y,X) → (Y,X,3) using an arbitrary channel→RGB mapping.

    channel_map : dict {channel_index: (R,G,B) coefficients}
        Example for swapping: {0: (0,1,0), 1: (1,0,1)}
    """
    if clip is None:
        lo, hi = np.percentile(frame_ctyx[list(channel_map)], (2, 99.8))
    else:
        lo, hi = clip

    # normalise selected channels only
    frame_norm = np.clip((frame_ctyx - lo) / (hi - lo), 0, 1)

    # build RGB
    rgb = np.zeros((*frame_ctyx.shape[1:], 3), dtype=np.float32)
    for ch, (r_coef, g_coef, b_coef) in channel_map.items():
        rgb[..., 0] += r_coef * frame_norm[ch]
        rgb[..., 1] += g_coef * frame_norm[ch]
        rgb[..., 2] += b_coef * frame_norm[ch]

    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

# ---------- colour-mapping helpers ------------------------------------------------
def _to_rgb(frame_ctyx: np.ndarray,
            channel_map: Mapping[int, Sequence[float]] = SWAP_MAP,
            clip: Optional[tuple[float, float]] = None) -> np.ndarray:
    """
    Convert (C,Y,X) → (Y,X,3) using:
      ch-0  → magenta  (R+B)
      ch-1  → green    (G)
    Additional channels are ignored.

    clip : (lo, hi)  → intensity range for min-max scaling
    """
    assert frame_ctyx.ndim == 3 and frame_ctyx.shape[0] >= 2, "need ≥2 channels"
    if clip is None:
        lo0, hi0 = np.percentile(frame_ctyx[0], (2, 99.8))
        lo1, hi1 = np.percentile(frame_ctyx[1], (2, 99.8))
        scale = [(lo0, hi0), (lo1, hi1)]
    else:
        scale = clip  # list of pairs or single pair

    rgb = np.zeros((*frame_ctyx.shape[1:], 3), dtype=np.float32)
    for ch, (r, g, b) in channel_map.items():
        if ch == 1:
            lo, hi = scale[ch] if isinstance(scale[0], tuple) else scale
            ch_norm = np.clip((frame_ctyx[ch] - lo) / (hi - lo), 0, 1)
            rgb[..., 0] += r * ch_norm
            rgb[..., 1] += g * ch_norm
            rgb[..., 2] += b * ch_norm

    return (rgb * 255).astype(np.uint8)

# ---------- main export routine ---------------------------------------------------
def export_timelapse(
    zarr_root: Path | str,
    experiment_date: str,
    wells: Sequence[int],
    *,
    save_as: Literal["mp4", "png", "tiff"] = "mp4",
    fps: int = 10,
    overwrite: bool = False,
    out_dir: Optional[Path | str] = None,
) -> None:
    """
    For each well in *wells* read `{date}_well####_z.zarr` and export a movie / image stack.

    Parameters
    ----------
    zarr_root : directory that contains well-level Zarrs
    experiment_date : e.g. '20250716'
    wells : iterable of well indices (ints)
    save_as : 'mp4', 'png', or 'tiff'
    fps : frames-per-second for mp4
    overwrite : replace existing outputs
    out_dir : where to write movies; default = zarr_root / 'movies'
    """
    zarr_root = Path(zarr_root)
    out_dir = Path(out_dir) if out_dir else zarr_root / "movies"
    out_dir.mkdir(exist_ok=True)

    for w in wells:
        z_name = f"{experiment_date}_well{w:04d}_z.zarr"
        z_path = zarr_root / z_name
        if not z_path.exists():
            print(f"[skip] {z_name} not found")
            continue

        out_stub = out_dir / z_name.replace("_z.zarr", "")
        if not overwrite and (
            (out_stub.with_suffix(".mp4")).exists()
            or (out_stub / "frame_0000.png").exists()
        ):
            print(f"[skip] output for well {w} already exists")
            continue

        # ---- open lazily via dask -------------------------------------------------
        data = da.from_zarr(z_path)     # shape (C,T,Y,X)
        c, t, y, x = data.shape
        if c < 2:
            print(f"[warn] well {w} has <2 channels, skipping")
            continue

        # ---- writer factory -------------------------------------------------------
        if save_as == "mp4":
            writer = imageio.get_writer(
                out_stub.with_suffix(".mp4"),
                fps=fps,
                codec="libx264",
                quality=8,
            )
        else:
            (out_stub).mkdir(exist_ok=True, parents=True)
            writer = None

        # ---- iterate over time ----------------------------------------------------
        for ti in tqdm(range(t), desc=f"well {w}"):
            frame = data[:, ti].compute()   # loads one (C,Y,X) chunk
            rgb = _to_rgb(frame, channel_map=SWAP_MAP)

            if save_as == "mp4":
                writer.append_data(rgb)
            else:
                ext = ".png" if save_as == "png" else ".tiff"
                fn = out_stub / f"frame_{ti:04d}{ext}"
                iio.imwrite(fn, rgb)

        if writer is not None:
            writer.close()

        print(f"[done] well {w} → {save_as.upper()} saved")


# ---------- CLI entry-point -------------------------------------------------------
if __name__ == "__main__":
    # example usage
    export_timelapse(
        zarr_root=r"Y:\projects\data\killi_tracker\built_data\zarr_image_files\20250731",
        experiment_date="20250716",
        wells=range(14),          # first four wells
        save_as="mp4",           # or 'png' / 'tiff'
        fps=3,
        overwrite=False,
        out_dir="E:\\Nick\\movies"  # optional, defaults to zarr_root / 'movies'
    )