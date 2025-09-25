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
# ---------- colour-mapping helpers ------------------------------------------------
def _to_rgb(frame_ctyx: np.ndarray,
            channel_map: Optional[Mapping[int, Sequence[float]]] = None,
            clip: Optional[tuple[float, float]] = None) -> np.ndarray:
    """
    Convert (C,Y,X) → (Y,X,3).

    If 1 channel → grayscale.
    If 2+ channels → use channel_map (default = magenta+green).
    """
    assert frame_ctyx.ndim == 3, "need (C,Y,X) array"

    c = frame_ctyx.shape[0]

    if c == 1:
        # grayscale mode
        if clip is None:
            lo, hi = np.percentile(frame_ctyx[0], (2, 99.8))
        else:
            lo, hi = clip
        ch_norm = np.clip((frame_ctyx[0] - lo) / (hi - lo), 0, 1)
        rgb = np.stack([ch_norm, ch_norm, ch_norm], axis=-1)

    else:
        # default magenta+green map if none supplied
        if channel_map is None:
            channel_map = {0: (1, 0, 1),   # ch-0 → magenta
                           1: (0, 1, 0)}   # ch-1 → green
        # get percentile scaling per channel
        if clip is None:
            scale = {ch: np.percentile(frame_ctyx[ch], (2, 99.8))
                     for ch in channel_map}
        else:
            # pass in either (lo,hi) for all or dict of per-channel
            scale = {ch: clip if isinstance(clip[0], (int,float)) else clip[ch]
                     for ch in channel_map}

        rgb = np.zeros((*frame_ctyx.shape[1:], 3), dtype=np.float32)
        for ch, (r, g, b) in channel_map.items():
            lo, hi = scale[ch]
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
        if c < 1:
            print(f"[warn] well {w} has no channels, skipping")
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
        zarr_root=r"/media/nick/cluster/projects/data/killi_tracker/built_data/zarr_image_files/20250621/",
        experiment_date="20250621",
        wells=range(32),          # first four wells
        save_as="mp4",           # or 'png' / 'tiff'
        fps=3,
        overwrite=False,
        out_dir="/media/nick/cluster/projects/data/killi_tracker/built_data/mips/20250621"  # optional, defaults to zarr_root / 'movies'
    )