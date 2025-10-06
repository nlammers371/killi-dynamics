import numpy as np
import zarr, healpy as hp
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import matplotlib
matplotlib.use("QtAgg")   # or "TkAgg" if Qt isn't installed
import matplotlib.pyplot as plt
from src.build_yx1.make_field_plots import laea_forward, laea_disk_mask, _laea_bins, _raster_mean


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
    pct=(2, 99),
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

    g = zarr.open_group(str(zarr_path), mode="r")
    vals = g[values_key]
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
    im = ax.imshow(
        img,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation="nearest"
    )

    im.set_alpha(np.isfinite(img).astype(float))

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
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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


if __name__ == '__main__':

    root = "/media/nick/cluster/projects/data/killi_tracker/"
    project_name = "20250621"
    projection_path = Path(root) / "output_data" / "sphere_projections" / project_name
    well_list = sorted(projection_path.glob("*.zarr"))
    well_ind = 3
    well_zarr_path = well_list[well_ind]
    values_key="max"
    t_index=30
    channel=0
    nside=None,

    plot_healpix_equalarea_frame(zarr_path=well_zarr_path,
                                 t=t_index,
                                 channel=channel,
                                 values_key="max",
                                 bins=None,
                                 smooth_fwhm_deg=8, #None,
                                 grid=True,
                                 vmin=0,
                                 vmax=1600,
                                 fov_deg=75,
                                 rescale_to_fill=True,
                                 save_path="test.png",
                                 cmap="turbo",)
