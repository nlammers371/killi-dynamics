#!/usr/bin/env python3
"""
healpix_to_movie_v2.py
Author: <you>
Description:
    Produce smooth, interpolated Healpix movies from zarr datasets.
    Adds temporal Gaussian smoothing and adjustable time resampling
    on top of your existing spatial smoothing and plotting workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import zarr
from pathlib import Path
import re
from typing import Union
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from functools import partial
from tqdm.contrib.concurrent import process_map

# ==============================================================
#  Low-level geometry + raster helpers
# ==============================================================

def laea_forward(theta, phi, hemisphere="north", phi0_deg=0.0):
    phi0 = np.deg2rad(phi0_deg)
    phi = (phi - phi0 + np.pi) % (2*np.pi) - np.pi
    if hemisphere == "north":
        r = np.sqrt(2.0) * np.sin(theta/2.0)
    elif hemisphere == "south":
        r = np.sqrt(2.0) * np.cos(theta/2.0)
    else:
        raise ValueError("hemisphere must be 'north' or 'south'")
    return r*np.cos(phi), r*np.sin(phi)


def laea_disk_mask(bins):
    xs = np.linspace(-1 + 1/bins, 1 - 1/bins, bins)
    ys = np.linspace(-1 + 1/bins, 1 - 1/bins, bins)
    Xc, Yc = np.meshgrid(xs, ys, indexing="ij")
    return (Xc*Xc + Yc*Yc) <= 1.0 + 1e-12


def _laea_bins(nside, bins, hemisphere="north", phi0_deg=0.0):
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=False)
    x, y = laea_forward(theta, phi, hemisphere=hemisphere, phi0_deg=phi0_deg)
    dx = dy = 2.0 / bins
    ix = np.floor((x + 1.0) / dx).astype(int)
    iy = np.floor((y + 1.0) / dy).astype(int)
    in_bounds = (ix >= 0) & (ix < bins) & (iy >= 0) & (iy < bins)
    in_disk = (x*x + y*y) <= 1.0 + 1e-12
    mask_in = in_bounds & in_disk
    ix[~mask_in] = -1
    iy[~mask_in] = -1
    disk_mask = laea_disk_mask(bins)
    return ix, iy, mask_in, disk_mask


def _raster_mean(values_1d, ix, iy, mask_in, bins, disk_mask):
    acc = np.zeros((bins, bins), dtype=float)
    cnt = np.zeros((bins, bins), dtype=np.int64)
    np.add.at(acc, (ix[mask_in], iy[mask_in]), values_1d[mask_in])
    np.add.at(cnt, (ix[mask_in], iy[mask_in]), 1)
    img = np.divide(acc, cnt, out=np.full_like(acc, np.nan), where=cnt > 0)
    img[~disk_mask] = np.nan
    return img


# ==============================================================
#  Core smoothing + interpolation
# ==============================================================

def _load_time_series_1ch(zarr_path, values_key="max", channel=0):
    """Return (T, Npix) array for a single channel."""
    g = zarr.open_group(str(zarr_path), mode="r") if values_key != "density" else None
    vals = g[values_key] if values_key != "density" else zarr.open_array(str(Path(zarr_path) / "density"), mode="r")
    if vals.ndim == 3:
        return np.asarray(vals[:, channel, :])
    elif vals.ndim == 2:
        return np.asarray(vals)
    else:
        raise ValueError("Expected (T, Npix) or (T, C, Npix)")


def temporal_smooth_and_resample(data_TN, dt_hours, target_dt_hours=None,
                                 temporal_sigma_hours=None, kind="linear"):
    """Gaussian smooth along time axis + resample to new time grid."""
    data = np.array(data_TN, copy=False)
    if temporal_sigma_hours and temporal_sigma_hours > 0:
        sigma_f = temporal_sigma_hours / dt_hours
        data = gaussian_filter1d(data, sigma=sigma_f, axis=0, mode="nearest")

    if target_dt_hours and target_dt_hours > 0 and not np.isclose(target_dt_hours, dt_hours):
        t_orig = np.arange(data.shape[0]) * dt_hours
        t_target = np.arange(0.0, t_orig[-1] + 1e-9, target_dt_hours)
        f = interp1d(t_orig, data, axis=0, kind=kind,
                     bounds_error=False, fill_value="extrapolate")
        data = f(t_target)
        return data, t_target
    else:
        t_orig = np.arange(data.shape[0]) * dt_hours
        return data, t_orig


# ==============================================================
#  Plotting from pre-smoothed single frame
# ==============================================================

def plot_healpix_equalarea_from_frame(
    frame_1d,
    hemisphere="south",
    phi0_deg=0.0,
    bins=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    pct=(1, 99.8),
    smooth_fwhm_deg=None,
    fov_deg=90.0,
    rescale_to_fill=False,
    dpi=600,
    save_path=None,
    show=False,
):
    frame = np.array(frame_1d, copy=False)
    nside = int(np.sqrt(frame.size / 12))

    # spatial smoothing
    if smooth_fwhm_deg is not None:
        frame = hp.smoothing(frame, fwhm=np.deg2rad(smooth_fwhm_deg))

    # rasterize
    if bins is None:
        bins = int(round(2 * nside))
    ix, iy, mask_in, disk_mask = _laea_bins(nside, bins, hemisphere, phi0_deg)
    theta, phi = hp.pix2ang(nside, np.arange(frame.size))
    x, y = laea_forward(theta, phi, hemisphere=hemisphere, phi0_deg=phi0_deg)
    r = np.sqrt(x**2 + y**2)
    r_max = np.sqrt(2) * np.sin(np.deg2rad(fov_deg) / 2)
    mask_in = mask_in & (r <= r_max)
    img = _raster_mean(frame, ix, iy, mask_in, bins, disk_mask)

    # color limits
    if  vmin is None or vmax is None:
        finite_vals = frame[np.isfinite(frame)]
        if finite_vals.size > 0:
            lo, hi = np.percentile(finite_vals, pct)
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        else:
            vmin, vmax = 0, 1

    # render
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    yy, xx = np.mgrid[-1:1:img.shape[0]*1j, -1:1:img.shape[1]*1j]
    rr = np.sqrt(xx**2 + yy**2)
    alpha = (rr <= 1.0).astype(float)

    im = ax.imshow(img, origin="lower", extent=[-1, 1, -1, 1],
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    im.set_alpha(alpha)
    ax.set_facecolor("none"); fig.patch.set_alpha(0.0)
    ax.set_aspect("equal")

    if (fov_deg < 90) and rescale_to_fill:
        r_crop = np.sqrt(2) * np.sin(np.deg2rad(fov_deg) / 2)
        zoom_factor = np.sqrt(2) / r_crop * 0.7 * 0.985
    else:
        zoom_factor = 1.0
    ax.set_xlim(-1/zoom_factor, 1/zoom_factor)
    ax.set_ylim(-1/zoom_factor, 1/zoom_factor)

    circle = plt.Circle((0, 0), 0.978/zoom_factor, facecolor='none',
                        edgecolor='black', lw=6.0, zorder=6)
    ax.add_patch(circle)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi, transparent=True)
    if show:
        plt.show()
    plt.close(fig)
    return img


# ==============================================================
#  Intensity scaling util
# ==============================================================

def get_intensity_bounds_from_last_frame(zarr_path, channel=0,
                                         values_key="max",
                                         smooth_fwhm_deg=None,
                                         pct=(1, 99.5)):
    g = zarr.open_group(str(zarr_path), mode="r")
    vals = g[values_key]
    if vals.ndim == 3:
        frame = np.array(vals[-1, channel])
    elif vals.ndim == 2:
        frame = np.array(vals[-1])
    else:
        raise ValueError("Expected (T, Npix) or (T, C, Npix)")

    if values_key == "density":
        N = frame.size; A = 4*np.pi*600**2; a = A/N
        frame = frame / a
    if smooth_fwhm_deg is not None:
        frame = hp.smoothing(frame, fwhm=np.deg2rad(smooth_fwhm_deg))
    finite_vals = frame[np.isfinite(frame) & (frame > 0)]
    if finite_vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite_vals, pct)
    return lo, hi

def make_well_plots(well,
                    project_name,
                    field_list,
                    out_root,
                    values_key,
                    dt_hours,
                    target_dt_hours,
                    temporal_sigma_hours,
                    interp_kind,
                    dpi,
                    plot_kwargs,
                    overwrite,
                    channel):

    zarr_path = [m for m in field_list if int(re.search(r"well(\d+)", str(m)).group(1)) == well][0]
    zarr_path = Path(zarr_path)
    out_path = out_root / f"{project_name}_well{well:04}_{values_key}"
    out_path.mkdir(parents=True, exist_ok=True)

    series_TN = _load_time_series_1ch(zarr_path, values_key=values_key, channel=channel)

    if values_key == "density":
        N = series_TN.shape[1];
        A = 4 * np.pi * 600 ** 2;
        a = A / N
        series_TN = series_TN / a

    if dt_hours is None:
        raise ValueError("Please supply dt_hours for temporal smoothing/interpolation.")
    series_out, t_out = temporal_smooth_and_resample(
        series_TN, dt_hours=dt_hours,
        target_dt_hours=target_dt_hours,
        temporal_sigma_hours=temporal_sigma_hours,
        kind=interp_kind,
    )

    if False: #scale_from_smoothed:
        vmin = vmax = None
    else:
        vmin, vmax = get_intensity_bounds_from_last_frame(
            zarr_path, channel,
            smooth_fwhm_deg=plot_kwargs.get("smooth_fwhm_deg"),
            values_key=values_key, pct=(1, 99.8)
        )
        vmax = 0.875 * vmax

    print(f"Rendering {series_out.shape[0]} frames for well {well:04d} ...")
    for t_idx in range(series_out.shape[0]):
        frame_file = out_path / f"{zarr_path.stem}_frame{t_idx:04d}.png"
        if not frame_file.exists() or overwrite:
            plot_healpix_equalarea_from_frame(
                series_out[t_idx],
                save_path=frame_file,
                vmin=vmin,
                vmax=vmax,
                **plot_kwargs,
                dpi=dpi,
            )
    print(f"✅ Saved {series_out.shape[0]} frames to {out_path}")
# ==============================================================
#  Main driver
# ==============================================================

def healpix_to_mp4_v2(
    root: Union[str, Path],
    project_name: str,
    wells=None,
    values_key="max",
    channel=0,
    nside=64,
    overwrite=False,
    plot_kwargs=None,
    dpi=600,
    # new:
    dt_hours=None,
    target_dt_hours=None,
    temporal_sigma_hours=None,
    interp_kind="linear",
    n_jobs=1,
):
    """
    Render Healpix zarr time series with temporal smoothing/interpolation.
    Produces PNGs (and optionally MP4 via external ffmpeg if desired).
    """
    plot_kwargs = plot_kwargs or {}
    zarr_root = Path(root) / "output_data" / "sphere_projections" / project_name
    field_list = sorted(list(Path(zarr_root).glob(f"*_{nside:04}.zarr")))
    out_root = Path(root) / "figures" / "sphere_projections_v2"
    out_root.mkdir(parents=True, exist_ok=True)

    if wells is None:
        wells = [int(re.search(r"well(\d+)", str(s)).group(1)) for s in field_list]

    plot_well = partial(make_well_plots,
                        project_name=project_name,
                        field_list=field_list,
                        out_root=out_root,
                        values_key=values_key,
                        dt_hours=dt_hours,
                        target_dt_hours=target_dt_hours,
                        temporal_sigma_hours=temporal_sigma_hours,
                        interp_kind=interp_kind,
                        dpi=dpi,
                        plot_kwargs=plot_kwargs,
                        overwrite=overwrite,
                        channel=channel)

    if n_jobs == 1:
        _ = [plot_well(a) for a in tqdm(wells, desc="Calculating density fields")]
    else:
        _ = process_map(
            plot_well,
            wells,
            max_workers=n_jobs,
            chunksize=1,
            desc="Calculating density fields",
        )



# ==============================================================
#  Usage example
# ==============================================================

if __name__ == "__main__":
    healpix_to_mp4_v2(
        root="/path/to/project/root",
        project_name="20250716",
        wells=[2, 8, 10],
        values_key="max",
        dt_hours=1.4,
        target_dt_hours=1.0,
        temporal_sigma_hours=2.0,
        plot_kwargs=dict(smooth_fwhm_deg=10, hemisphere="south", cmap="viridis"),
        mode="png",
        overwrite=True,
    )
