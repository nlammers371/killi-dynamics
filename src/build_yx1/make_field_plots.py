import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import zarr, healpy as hp
from pathlib import Path
import re
from typing import Union
from tqdm import tqdm


def get_intensity_bounds_from_last_frame(
        zarr_path,
        channel=0,
        values_key="max",
        smooth_fwhm_deg=None,
        pct=(1, 99)
):
    """
    Estimate vmin/vmax for a given well based on the last timepoint.
    This helps keep color scaling stable over time while adapting to
    per-well intensity differences.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the zarr group containing the field values.
    channel : int
        Channel index if data are (T, C, Npix).
    values_key : str
        Dataset key (e.g. 'max', 'mean', etc.)
    pct : tuple
        Percentile range for scaling (e.g., (2, 99)).

    Returns
    -------
    vmin, vmax : float
        Suggested color scale bounds.
    """
    g = zarr.open_group(str(zarr_path), mode="r")
    vals = g[values_key]
    if vals.ndim == 3:
        frame = np.array(vals[-1, channel])
    elif vals.ndim == 2:
        frame = np.array(vals[-1])
    else:
        raise ValueError("Expected (T, Npix) or (T, C, Npix) array.")

    if values_key == "density":
        # rescale to density per mm^2
        N = frame.size
        A = 4 * np.pi * 600**2
        a = A  / N
        frame = frame / a

    if smooth_fwhm_deg is not None:
        frame = hp.smoothing(frame, fwhm=np.deg2rad(smooth_fwhm_deg))

    finite_vals = frame[np.isfinite(frame) & (frame > 0)]
    if finite_vals.size == 0:
        return 0.0, 1.0

    lo, hi = np.percentile(finite_vals, pct)

    if values_key != "density":
        # round to nearest 100 cells/mm^2
        lo = np.round(lo / 100) * 100 # round to nearest 10
        lo = np.clip(lo, 0, 400)
        hi = np.round(hi / 100) * 100
        hi = np.clip(hi, 600, 3000)
    else:
        lo = np.floor(lo * 1e4) *1e-4
        hi = np.ceil(hi * 1e4) *1e-4

    return lo, hi

# -------------------------------
# LAEA helpers
# -------------------------------
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


# -------------------------------
# Projection binning + raster
# -------------------------------
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


# -------------------------------
# Single-frame plotting
# -------------------------------
def plot_healpix_equalarea_frame(
    zarr_path,
    t=0,
    channel=0,
    values_key="max",
    bins=None,
    phi0_deg=0.0,
    hemisphere="south",
    cmap="viridis",
    vmin=None,
    vmax=None,
    pct=(1, 99),
    show=True,
    save_path=None,
    smooth_fwhm_deg=None,
    fov_deg=90.0,
    rescale_to_fill=False,
    dpi=600
):
    """
    Plot a single timepoint of a HEALPix scalar field in equal-area projection.

    Parameters
    ----------
    smooth_fwhm_deg : float or None
        Optional Gaussian smoothing FWHM in degrees (uses healpy.smoothing).
    fov_deg : float
        Radial field of view in degrees (default 90 = full hemisphere).
    grid : bool
        If True, overlay azimuthal grid lines.
    """

    if values_key != "density":
        g = zarr.open_group(str(zarr_path), mode="r")
        vals = g[values_key]
    else:
        vals = zarr.open_array(str(zarr_path / "density"), mode="r")
        # rescale to debsity per mm^2
        N = vals[0].size
        A = 4 * np.pi * 600**2
        a = A  / N

    if vals.ndim == 3:
        frame = np.array(vals[t, channel])
    elif vals.ndim == 2:
        frame = np.array(vals[t])
    else:
        raise ValueError("Expected (T, Npix) or (T, C, Npix)")

    nside = int(np.sqrt(frame.size / 12))

    # (i) Optional smoothing
    if smooth_fwhm_deg is not None:
        frame = hp.smoothing(frame, fwhm=np.deg2rad(smooth_fwhm_deg))

    if values_key == "density":
        frame = frame / a

    # (ii) Binning to raster
    if bins is None:
        bins = int(round(2 * nside))
    ix, iy, mask_in, disk_mask = _laea_bins(nside, bins, hemisphere, phi0_deg)

    # (ii) Optional FOV cro
    theta, phi = hp.pix2ang(nside, np.arange(frame.size))
    x, y = laea_forward(theta, phi, hemisphere=hemisphere, phi0_deg=phi0_deg)
    r = np.sqrt(x**2 + y**2)
    r_max = np.sqrt(2) * np.sin(np.deg2rad(fov_deg) / 2)
    mask_in = mask_in & (r <= r_max)

    img = _raster_mean(frame, ix, iy, mask_in, bins, disk_mask)

    # Color scaling
    if vmin is None or vmax is None:
        finite_vals = frame[np.isfinite(frame)]
        if finite_vals.size > 0:
            lo, hi = np.percentile(finite_vals, pct)
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        else:
            vmin, vmax = 0, 1

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    # --- make alpha mask circular ---
    yy, xx = np.mgrid[-1:1:img.shape[0] * 1j, -1:1:img.shape[1] * 1j]
    rr = np.sqrt(xx ** 2 + yy ** 2)
    alpha = (rr <= 1.0).astype(float)  # 1 inside unit circle, 0 outside

    im = ax.imshow(
        img,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    im.set_alpha(alpha)

    # make both axes and figure transparent
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    # Keep equal scaling so the circle stays circular
    ax.set_aspect("equal")

    # --- ZOOM logic ---
    # zoom_factor < 1 shows more of the frame (smaller disk)
    # zoom_factor > 1 crops in symmetrically, keeping circle shape
    if (fov_deg < 90) and rescale_to_fill:
        # geometric crop radius for given FOV
        r_crop = np.sqrt(2) * np.sin(np.deg2rad(fov_deg) / 2)
        # scale so the chosen FOV expands to fill more of the plot
        zoom_factor = np.sqrt(2) / r_crop * 0.7 * 0.985  # tweak multiplier 0.7–1.0 for taste
    else:
        zoom_factor = 1.0

    ax.set_xlim(-1 / zoom_factor, 1 / zoom_factor)
    ax.set_ylim(-1 / zoom_factor, 1 / zoom_factor)

    # --- background + frame ---
    ax.set_facecolor("none")  # transparent corners
    fig.patch.set_facecolor("white")  # white figure background (shows in saved image)
    ax.set_xticks([])
    ax.set_yticks([])
    # cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    circle = plt.Circle(
        (0, 0), 0.978 / zoom_factor,
        facecolor='none', edgecolor='black', lw=6.0, zorder=6
    )
    ax.add_patch(circle)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # (iv) Optional rescaling to fill
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)

    return img

# -------------------------------
# MP4 wrapper
# -------------------------------
def healpix_to_mp4(
    root: Union[str, Path],
    project_name: str,
    wells=None,
    values_key="max",
    channel=0,
    fps=12,
    nside=64,
    mode="png",               # folder for temporary or final PNGs
    overwrite=False,    # remove PNGs after MP4 creation
    plot_kwargs=None,
    dpi=600,
):
    """
    Iterate over frames in a HEALPix zarr dataset, calling
    `plot_healpix_equalarea_frame` for each, and either:
      (a) save PNGs, or
      (b) save PNGs and compile them into an MP4.

    Parameters
    ----------
    zarr_path : str or Path
        Zarr group containing [T, (C), Npix] dataset.
    out_path : str or Path
        Output path: .mp4 if mode='mp4', directory if mode='png'.
    mode : {"mp4", "png"}
        Output format.
    frame_dir : str or Path, optional
        Directory to store PNG frames (created if needed).
    fps : int
        Frame rate for MP4.
    overwrite : bool
        Overwrite existing files.
    cleanup : bool
        Delete temporary PNGs after MP4 creation.
    plot_kwargs : dict
        Keyword args passed to `plot_healpix_equalarea_frame`.
    """
    plot_kwargs = plot_kwargs or {}

    zarr_path = Path(root) / "output_data" / "sphere_projections" / project_name
    field_list = sorted(list(Path(zarr_path).glob(f"*_{nside:04}.zarr")))

    # output path
    out_root = Path(root) / "figures" / "sphere_projections"
    out_root.mkdir(parents=True, exist_ok=True)

    if wells is None:
        wells = [int(re.search(r"well(\d+)", str(s)).group(1)) for s in field_list]

    for well in tqdm(wells, desc="Rendering frames"):

        zarr_path = [m for m in field_list if int(re.search(r"well(\d+)", str(m)).group(1)) == well][0]
        zarr_path = Path(zarr_path)
        out_path = out_root / f"{project_name}_well{well:04}_{values_key}"
        out_path.mkdir(parents=True, exist_ok=True)

        g = zarr.open_group(str(zarr_path), mode="r")
        vals = g[values_key]
        T = vals.shape[0]

        vmin, vmax = get_intensity_bounds_from_last_frame(zarr_path, channel, smooth_fwhm_deg=plot_kwargs["smooth_fwhm_deg"],
                                                          values_key=values_key, pct=(1, 99.5))

        # determine where to write PNGs
        # frame_dir = out_path.with_suffix("") / "frames"
        # frame_dir = Path(frame_dir)
        # frame_dir.mkdir(parents=True, exist_ok=True)

        # loop over frames -> generate PNGs
        print(f"Rendering {T} frames to {out_path} ...")
        for t in reversed(range(T)):
            frame_file = out_path / f"{zarr_path.stem}_frame{t:04d}.png"
            if not frame_file.exists() or overwrite:
                plot_healpix_equalarea_frame(
                    zarr_path,
                    t=t,
                    channel=channel,
                    dpi=dpi,
                    values_key=values_key,
                    show=False,
                    save_path=frame_file,
                    vmin=vmin,
                    vmax=vmax,
                    **plot_kwargs,
                )

        # if PNG mode, we’re done
        if mode == "png":
            print(f"✅ Saved {T} PNG frames to {out_path}")
            # return

        # # else compile into MP4
        # print(f"Compiling {T} frames into {out_path} ...")
        # fig, ax = plt.subplots(figsize=(6, 6))
        # writer = FFMpegWriter(  fps=fps,
        #                         codec="libx264",
        #                         extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        #                       )
        # mp4_path = out_root / f"{project_name}_well{well:04}.mp4"
        # with writer.saving(fig, str(mp4_path), dpi=dpi):
        #     for t in range(T):
        #         img = plt.imread(frame_dir / f"{zarr_path.stem}_frame{t:04d}.png")
        #         ax.clear()
        #         ax.imshow(img)
        #         ax.axis("off")
        #         writer.grab_frame()
        # plt.close(fig)
        # print(f"✅ Saved MP4 to {mp4_path}")

        # cleanup
        # if cleanup:
        #     for f in frame_dir.glob("*.png"):
        #         f.unlink()
            # try:
            #     frame_dir.rmdir()
            # except OSError:
            #     pass
