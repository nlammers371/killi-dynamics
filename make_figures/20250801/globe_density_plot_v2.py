import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from tqdm import tqdm
import matplotlib as mpl

def rotate_lonlat(lon, lat, yaw=0.0, pitch=0.0, roll=0.0, degrees=True):
    """
    Rotate spherical points by intrinsic ZYX (yaw->Z, pitch->Y, roll->X).
    - yaw   : spin around vertical (z) axis  (azimuthal)
    - pitch : tilt forward/back around y axis (moves poles)
    - roll  : tilt around x axis
    Returns new (lon_rot, lat_rot).
    """
    if degrees:
        yaw  = np.deg2rad(yaw)
        pitch= np.deg2rad(pitch)
        roll = np.deg2rad(roll)

    # to Cartesian
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    v = np.stack([x, y, z], axis=-1)

    # rotation matrices (intrinsic ZYX)
    cz, sz = np.cos(yaw),   np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll),  np.sin(roll)

    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])

    R = Rz @ Ry @ Rx  # Z then Y then X

    v_rot = v @ R.T
    x2, y2, z2 = v_rot[...,0], v_rot[...,1], v_rot[...,2]

    lat2 = np.rad2deg(np.arcsin(np.clip(z2, -1, 1)))
    lon2 = np.rad2deg(np.arctan2(y2, x2))
    return lon2, lat2

# ------------------------------------------------------------
# 0) CONFIG
# ------------------------------------------------------------
start_frame_max    = 0
phi_shift_manual   = 0 #60 * np.pi / 180
lon_bins           = 360          # horizontal resolution
lat_bins           = 180          # vertical resolution (equal-area in sin(lat))
time_window        = 10           # frames on each side
sigma_lon_bins     = 5.0          # Gaussian smooth std in *bins* (lon)
sigma_lat_bins     = 3.5          # Gaussian smooth std in *bins* (lat)
cmap_name          = "inferno"
vmin, vmax         = None, None   # let Matplotlib pick; set numbers if you want fixed colors
draw_contours      = True
contour_levels     = 10           # or a list/array for exact levels
proj               = ccrs.Robinson()
dpi_save           = 600
DARK_MODE = True
# Map "north hemisphere" coverage → "east hemisphere" coverage
remap_north_to_east = False
remap_north_to_west = False  # alternative, if the sign is flipped
shift_theta = False
# shift theta to make the north pole at +90° (default) or -90°
if DARK_MODE:
    mpl.rcParams.update({
        # backgrounds
        "figure.facecolor":  "black",
        "axes.facecolor":    "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black",
        # text & spines
        "text.color":        "white",
        "axes.edgecolor":    "white",
        "axes.labelcolor":   "white",
        "xtick.color":       "white",
        "ytick.color":       "white",
        # lines/grids
        "grid.color":        "0.5",
        "grid.alpha":        0.3,
    })

# 0) Set track paths
for i in [0, 1]:
    if i ==0:
        root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
        project_name = "20250419_BC1-NLSMSC"
        tracking_config = "tracking_20250328_redux"
        fig_path = os.path.join(root, "figures" , project_name)
        os.makedirs(fig_path, exist_ok=True)

        start_i = 0
        stop_i = 614
        suffix = ""
        scale_vec = np.asarray([3.0, 0.85, 0.85])
        flip_lat = True
        t_start = 56


# pitch = 60  # spin east by +30°
    else:
        project_name = "20250311_LCP1-NLSMSC"
        tracking_config = "tracking_20250328_redux"
        fig_path = os.path.join(root, "figures" , project_name)
        os.makedirs(fig_path, exist_ok=True)

        start_i = 0
        stop_i = 2339
        suffix = "_cb"
        scale_vec = np.asarray([3.0, 1, 1])
        flip_lat = False
        phi_shift_manual = 60 * np.pi / 180
        t_start = 26
        pitch = 0  # spin east by +0°

# project_name = "20240611_NLS-Kikume_24hpf_side2" #"20250419_BC1-NLSMSC"
# tracking_config = "tracking_jordao_20240918" #"tracking_20250328_redux"
# fig_path = os.path.join(root, "figures" , project_name)
# os.makedirs(fig_path, exist_ok=True)
#
# start_i = 0
# stop_i = 1600 # 614
# suffix = ""
# scale_vec = np.asarray([3.0, 1, 1])
# flip_lat = False
# phi_shift_manual = -60 * np.pi / 180
# shift_theta = True
# remap_north_to_west = True

    # load track df
    nls_track_path = os.path.join(root, "tracking", project_name, tracking_config, "well0000",  f"track_{start_i:04}_{stop_i:04}{suffix}")
    try:
        nls_tracks_df = pd.read_csv(os.path.join(nls_track_path, "tracks.csv"))
    except:
        nls_tracks_df = pd.read_csv(os.path.join(nls_track_path, "tracks_fluo.csv"))

    # scale the coordinates
    nls_tracks_df["z_scaled"] = nls_tracks_df["z"] * scale_vec[0]
    nls_tracks_df["x"] = nls_tracks_df["x"] * scale_vec[2]
    nls_tracks_df["y"] = nls_tracks_df["y"] * scale_vec[1]

    # ------------------------------------------------------------
    # 1) INPUTS (expects your existing DataFrames/paths)
    # ------------------------------------------------------------
    # nls_track_path, fig_path, nls_tracks_df must exist in your environment
    sphere_df    = pd.read_csv(os.path.join(nls_track_path, "sphere_fit.csv"))
    # nls_tracks_df = nls_tracks_df.loc[nls_tracks_df["track_class"]==0, ["t", "x", "y", "z_scaled"]].copy()

    # ------------------------------------------------------------
    # 2) ORIENT COORDS: align z-axis to your dome/up-vector, apply phi shift
    # ------------------------------------------------------------
    # sphere center and radius at t=0
    sphere_center = (sphere_df.loc[sphere_df["t"]==0, ["xs","ys","zs"]]
                     .iloc[0].to_numpy())
    # sphere_radius = sphere_df.loc[sphere_df["t"]==0, ["r"]].iloc[0].to_numpy()[0]  # optional

    # up-vector from center to early center-of-mass
    start_filter = nls_tracks_df["t"] <= start_frame_max
    deep_cm = (nls_tracks_df.loc[start_filter, ["x","y","z_scaled"]]
               .to_numpy().mean(axis=0))
    v = deep_cm - sphere_center
    v = v / np.linalg.norm(v)

    # rotation taking +z to v
    rot, _ = R.align_vectors([[0,0,1]], [v])
    Rmat = rot.as_matrix()

    # rotate all points (center first)
    pts     = nls_tracks_df[["x","y","z_scaled"]].to_numpy() - sphere_center
    pts_rot = (Rmat @ pts.T).T


    # if remap_north_to_east:
    #     # Rotate -90° about the Y-axis: z -> x (north becomes east)
    #     R_extra = R.from_euler('y', -90, degrees=True).as_matrix()
    #     pts_rot = (R_extra @ pts_rot.T).T
    #
    # elif remap_north_to_west:
    #     # Rotate +90° about the Y-axis: z -> -x (north becomes west)
    #     R_extra = R.from_euler('y', +90, degrees=True).as_matrix()
    #     pts_rot = (R_extra @ pts_rot.T).T

    # spherical coords (theta: 0..pi from +z, phi: -pi..pi)
    x, y, z = pts_rot[:,0], pts_rot[:,1], pts_rot[:,2]
    r       = np.linalg.norm(pts_rot, axis=1)
    theta   = np.arccos(np.clip(z/r, -1, 1))
    phi     = np.arctan2(y, x) + phi_shift_manual
    phi     = (phi + np.pi) % (2*np.pi) - np.pi  # wrap to (-pi, pi]



    # ---- your pipeline ----
    # after you build pts_rot (centered + aligned to your biological 'up' direction):
    # YOU SAID: y should be up/down  → pick up_axis='y'
    # Keep east_axis='x' unless you prefer a different 0° reference
    # r, theta, phi = spherical_with_axis(pts_rot, up_axis='y', east_axis='x')

    # optional tweaks
    phi += phi_shift_manual
    phi = (phi + np.pi) % (2*np.pi) - np.pi

    lon = np.degrees(phi)
    lat = 90.0 - np.degrees(theta)

    # lon, lat = rotate_lonlat(lon, lat, pitch=pitch)

    # # convert for mapping
    # lon = np.degrees(phi)
    # # Shift "north pole" up by 90°
    # if shift_theta:
    #     theta_shifted = theta - np.pi / 3  # or + np.pi/2 to shift down
    #
    #     # Wrap back into [0, π]
    #     theta_shifted = (theta_shifted + np.pi) % (np.pi)
    #
    #     lat = 90.0 - np.degrees(theta_shifted)
    # else:
    # lat = 90.0 - np.degrees(theta)
    # [-180, 180]

    if flip_lat:# [+90 at pole ... -90]
        lat = -lat
    t_vec = nls_tracks_df["t"].to_numpy()

    # ------------------------------------------------------------
    # 3) EQUAL-AREA GRID CONSTRUCTION
    #    We bin uniformly in lon, and uniformly in s = sin(lat_rad).
    #    Each (lon_bin × s_bin) cell has area Δλ * Δs on the unit sphere.
    # ------------------------------------------------------------
    lon_edges = np.linspace(-180.0, 180.0, lon_bins + 1, endpoint=True)

    # equal-area in latitude: uniform bins in s = sin(lat_rad)
    lat_rad_edges = np.arcsin(np.linspace(-1.0, 1.0, lat_bins + 1))
    lat_edges     = np.degrees(lat_rad_edges)

    # Precompute cell area (unit sphere): area_ij = Δλ(rad) * Δs
    dlam = np.deg2rad(np.diff(lon_edges))               # shape (lon_bins,)
    ds   = np.diff(np.sin(np.deg2rad(lat_edges)))       # shape (lat_bins,)
    cell_area = np.outer(ds, dlam)                      # shape (lat_bins, lon_bins); constant per row/col
    # (Optionally multiply by sphere_radius**2 if you want physical area)

    # Helper to wrap-smooth in lon and reflect at poles
    def smooth_cyclic_latlon(grid, sig_lon, sig_lat):
        """Gaussian smooth with wrap in lon (axis=1) and reflect at poles (axis=0)."""
        # pad lon by wrap
        pad = 3 * int(np.ceil(sig_lon))  # enough to avoid edge artifacts
        g_ext = np.concatenate([grid[:, -pad:], grid, grid[:, :pad]], axis=1)
        # smooth with reflect on lat (axis 0) and nearest on lon (axis 1 is fine after wrap)
        g_s = gaussian_filter(g_ext, sigma=(sig_lat, sig_lon), mode=("reflect", "nearest"))
        return g_s[:, pad:-pad]

    # ------------------------------------------------------------
    # 4) OUTPUT FOLDER
    # ------------------------------------------------------------
    hm_path = os.path.join(fig_path, "density_equal_area", "")
    os.makedirs(hm_path, exist_ok=True)

    # fixed cartopy extent for Robinson
    xlim = proj.x_limits
    ylim = proj.y_limits
    extent = (*xlim, *ylim)

    # ------------------------------------------------------------
    # 5) MAIN LOOP: accumulate counts in a moving time window, convert to density,
    #               smooth, and plot (optionally with contours)
    # ------------------------------------------------------------

    # Build frame list and reverse it (densest is first)
    frames = np.arange(0, int(t_vec.max()) + 1, time_window)[::-1]

    # Choose a robust percentile for the upper cap (tweak as you like)
    ROBUST_VMAX_Q = 0.99    # 99.5% (use 1.0 for absolute max)
    LOCK_SCALE_FROM_FIRST = True

    vmin_fixed = 0.0         # keep 0 for density
    vmax_fixed = None        # will be set on first reverse frame

    for i, frame in enumerate(frames):
        t_filter = (t_vec >= frame - time_window) & (t_vec <= frame + time_window)
        n_frames = len(np.unique(t_vec[t_filter])) if np.any(t_filter) else 1

        lon_f = lon[t_filter]
        lat_f = lat[t_filter]

        counts, y_edges, x_edges = np.histogram2d(lat_f, lon_f,
                                                  bins=[lat_edges, lon_edges])
        density = counts / (cell_area * max(n_frames, 1))
        density_s = smooth_cyclic_latlon(density, sigma_lon_bins, sigma_lat_bins)

        # Lock color scale from the first (latest) frame
        if i == 0 and LOCK_SCALE_FROM_FIRST:
            vmax_fixed = np.quantile(density_s, ROBUST_VMAX_Q)

        # centers/edges for plotting
        lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        LonC, LatC  = np.meshgrid(lon_centers, lat_centers)
        LonE, LatE  = np.meshgrid(lon_edges, lat_edges)

        # stage label unchanged
        stage = frame * 1.5 / 60 + t_start  # hpf

        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(projection=proj))

        hm = ax.pcolormesh(LonE, LatE, density_s,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap_name,
                           vmin=vmin_fixed,
                           vmax=vmax_fixed,
                           shading="auto")

        if draw_contours:
            cs = ax.contour(LonC, LatC, density_s,
                            levels=contour_levels,
                            transform=ccrs.PlateCarree(),
                            linewidths=0.8, alpha=0.9, colors="white")
            # ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

        ax.set_global()

        # Dark-theme grid + colorbar styling (from section i)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3)
        gl.xlabel_style = {"color":"white"}
        gl.ylabel_style = {"color":"white"}
        gl.top_labels = False
        gl.right_labels = False

        cbar = fig.colorbar(hm, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("cells per steradian per frame", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")
        cbar.outline.set_edgecolor("white")

        ax.set_title(f"Deep cell density on embryonic surface ({stage:.2f} hpf)")

        plt.tight_layout()
        out_path = os.path.join(hm_path, f"{project_name}_deep_cell_density_f{frame:04}.png")
        fig.savefig(out_path, dpi=dpi_save, bbox_inches="tight")
        plt.close(fig)