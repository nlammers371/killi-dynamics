import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import zarr, healpy as hp
from pathlib import Path

# -------------------------------
# LAEA helpers
# -------------------------------
def laea_forward(theta, phi, hemisphere="north", phi0_deg=0.0):
    """
    Lambert azimuthal equal-area projection.

    Parameters
    ----------
    theta : array (rad), colatitude [0, pi]
    phi   : array (rad), longitude [0, 2pi)
    hemisphere : 'north' or 'south'
        Which pole is at the disk center
    phi0_deg : float
        Recenter longitude (degrees)

    Returns
    -------
    x, y : arrays, coords in unit disk (equator -> r=1)
    """
    # recenter longitude
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
    """Precompute a circular mask for plotting the unit disk."""
    xs = np.linspace(-1 + 1/bins, 1 - 1/bins, bins)
    ys = np.linspace(-1 + 1/bins, 1 - 1/bins, bins)
    Xc, Yc = np.meshgrid(xs, ys, indexing="ij")
    return (Xc*Xc + Yc*Yc) <= 1.0 + 1e-12

# -------------------------------
# Projection binning
# -------------------------------
def _laea_bins(nside, bins, hemisphere="north", phi0_deg=0.0):
    """Precompute bin indices for all HEALPix pixels in LAEA projection."""
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
    img = np.divide(acc, cnt, out=np.full_like(acc, np.nan), where=cnt>0)
    img[~disk_mask] = np.nan
    return img

# -------------------------------
# Main function
# -------------------------------
def healpix_to_equalarea_mp4(
    well_zarr_path,
    out_mp4,
    values_key="raw",
    channel=0,
    bins=64,
    fps=12,
    phi0_deg=0.0,
    hemisphere="south",
    cmap="viridis",
    vmin=None, vmax=None,
    pct=(2, 98),
):
    g = zarr.open_group(str(well_zarr_path), mode="r")
    vals = g[values_key]           # (T, C, Npix) or (T, Npix)
    T = vals.shape[0]
    nval = vals.shape[-1]
    nside = int(np.sqrt(nval/12))

    # select channel
    if vals.ndim == 3:
        get_frame = lambda t: np.array(vals[t, channel])
        title_suffix = f"  ch={channel}"
    elif vals.ndim == 2:
        get_frame = lambda t: np.array(vals[t])
        title_suffix = ""
    else:
        raise ValueError("Expected vals of shape (T,Npix) or (T,C,Npix).")

    # projection bins
    ix, iy, mask_in, disk_mask = _laea_bins(nside, bins, hemisphere=hemisphere, phi0_deg=phi0_deg)

    # color scale
    if vmin is None or vmax is None:
        idx = np.linspace(0, T-1, num=min(T, 100), dtype=int)
        samp = np.concatenate([get_frame(t)[np.isfinite(get_frame(t))]
                               for t in idx if np.isfinite(get_frame(t)).any()])
        if samp.size:
            lo, hi = np.percentile(samp, pct)
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        else:
            vmin, vmax = 0.0, 1.0

    # figure + writer
    fig, ax = plt.subplots(figsize=(6,6), dpi=120)
    im = ax.imshow(np.zeros((bins,bins)), origin="lower", extent=[-1,1,-1,1],
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    time_arr = np.array(g["time"]) if "time" in g else None
    title = ax.set_title(f"{Path(well_zarr_path).name}{title_suffix}  t=0")

    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    with writer.saving(fig, str(out_mp4), 120):
        for t in range(T):
            frame = get_frame(t)
            img = _raster_mean(frame, ix, iy, mask_in, bins, disk_mask)
            im.set_data(img)
            if time_arr is not None:
                title.set_text(f"{Path(well_zarr_path).name}{title_suffix}  t={time_arr[t]}")
            else:
                title.set_text(f"{Path(well_zarr_path).name}{title_suffix}  t={t}")
            writer.grab_frame()
    plt.close(fig)
