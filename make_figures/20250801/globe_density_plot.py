# get spherical coordinates oriented relative to high dome position
from scipy.spatial.transform import Rotation as R
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from tqdm import tqdm


sphere_df = pd.read_csv(os.path.join(nls_track_path, "sphere_fit.csv"))
start_frame_max = 25
phi_shift_manual = 60 / 180 * np.pi

deep_tracks_df = nls_tracks_df.loc[nls_tracks_df["track_class"]==0, ["t", "x", "y", "z_scaled"]].copy()

start_filter = deep_tracks_df["t"] <= start_frame_max
start_indices = np.where(start_filter)[0]

# 1) pull out your sphere center in (x,y,z) order
sphere_center = sphere_df .loc[sphere_df["t"]==0, ["xs","ys","zs"]] \
                          .iloc[0] \
                          .to_numpy()    # [x0,y0,z0]
sphere_radius = sphere_df .loc[sphere_df["t"]==0, ["r"]] \
                          .iloc[0] \
                          .to_numpy()

# 2) center‐of‐mass also in (x,y,z)
start_filter = deep_tracks_df["t"] <= start_frame_max
deep_cm = ( deep_tracks_df
            .loc[start_filter, ["x","y","z_scaled"]]
            .to_numpy()
            .mean(axis=0) )                  # [x̄,ȳ,z̄]

# 3) up‐vector = direction from sphere_center → deep_cm
orientation_vec = deep_cm - sphere_center
v = orientation_vec / np.linalg.norm(orientation_vec)

rot, _ = R.align_vectors([[0,0,1]], [v])
# Note: align_vectors(A,B) finds R so that R @ A[i] ≈ B[i],
# so here we align the *z‑axis* to your v.
# If you prefer the opposite convention, swap the lists.

Rmat = rot.as_matrix()

# 3) apply R to all points (shift first)
pts = deep_tracks_df.loc[:, ["x","y","z_scaled"]].to_numpy() - sphere_center
pts_rot = (Rmat @ pts.T).T   # now “pole” is z

# 4) compute standard spherical coords
x, y, z = pts_rot[:,0], pts_rot[:,1], pts_rot[:,2]
r   = np.linalg.norm(pts_rot, axis=1)
theta = np.arccos(np.clip(z/r, -1, 1))   # 0…π
phi = np.arctan2(y, x)                 # –π…π
phi += phi_shift_manual
phi_wrapped = (phi + np.pi) % (2*np.pi) - np.pi
deep_tracks_df[["r","theta","phi"]] = np.column_stack([r, theta, phi_wrapped])

r, theta, phi = deep_tracks_df["r"].to_numpy(), deep_tracks_df["theta"].to_numpy(), deep_tracks_df["phi"].to_numpy()
# deep_tracks_df[["r","theta","phi"]] = np.column_stack([r, theta, phi])

# ------------------------------------------------------------
# 1) Example: N random points on (or near) the unit sphere
t_vec = deep_tracks_df["t"].to_numpy()

lon = np.degrees(phi)                        # 0–360
lat = 90 - np.degrees(theta)                       # 90 at pole … -90

# ------------------------------------------------------------
# 2) Robinson projection with Cartopy
proj = ccrs.Robinson()
xy = proj.transform_points(ccrs.PlateCarree(), lon, lat)
x_proj, y_proj = xy[:,0], xy[:,1]

from scipy.spatial import cKDTree
import matplotlib as mpl
from cycler import cycler

mpl.rcParams.update({
    # --------  backgrounds  --------
    "figure.facecolor":  "black",
    "axes.facecolor":    "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black",

    # --------  text / lines  -------
    "text.color":        "white",
    "axes.edgecolor":    "white",
    "axes.labelcolor":   "white",
    "xtick.color":       "white",
    "ytick.color":       "white",
    "grid.color":        "0.5",

    # brighter default colour‑cycle so traces stay visible
    "axes.prop_cycle":   cycler(color=plt.cm.tab10.colors)
})

r, theta, phi = deep_tracks_df["r"].to_numpy(), deep_tracks_df["theta"].to_numpy(), deep_tracks_df["phi"].to_numpy()
# deep_tracks_df[["r","theta","phi"]] = np.column_stack([r, theta, phi])

# ------------------------------------------------------------
# 1) Example: N random points on (or near) the unit sphere
t_vec = deep_tracks_df["t"].to_numpy()

lon = np.degrees(phi)                        # 0–360
lat = 90 - np.degrees(theta)                       # 90 at pole … -90

# ------------------------------------------------------------
# 2) Robinson projection with Cartopy
proj = ccrs.Robinson()
xy = proj.transform_points(ccrs.PlateCarree(), lon, lat)
x_proj, y_proj = xy[:,0], xy[:,1]



# proj = ccrs.Robinson()
nbins = 75
t_window = 25
hm_path = os.path.join(fig_path, "density_hexbin", "")
os.makedirs(hm_path, exist_ok=True)


# ------------------------------------------------------------------
# 2.  CONSTANT MAP EXTENT (Robinson globe bounds)
proj   = ccrs.Robinson()
xlim   = proj.x_limits               # (-1.68e7 , +1.68e7)
ylim   = proj.y_limits               # (-8.63e6 , +8.63e6)
extent = (*xlim, *ylim)              # (xmin, xmax, ymin, ymax)

# ------------------------------------------------------------------
# 3.  BUILD OUTPUT FOLDER
# hm_path = Path(fig_path) / "density_hexbin"
# hm_path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 4.  SELECT FRAME WINDOW & NORMALISE
for frame in tqdm(range(0, 2339, 25)):

    t_filter   = (t_vec >= frame - t_window) & (t_vec <= frame + t_window)
    n_frames   = len(np.unique(t_vec[t_filter]))   # number of frames in window
    xp_frame   = x_proj[t_filter]
    yp_frame   = y_proj[t_filter]

    # ------------------------------------------------------------------
    # 5.  INITIAL HEXBIN  (in axes coords, no projection needed here)
    #     We need the bin centres to build the KD‑tree.
    fig_tmp, ax_tmp = plt.subplots()
    hb_tmp = ax_tmp.hexbin(
            xp_frame, yp_frame,
            gridsize=nbins,
            extent=extent,
            bins=None,                 # raw counts
            mincnt=1,
            cmap="hot_r"
    )
    plt.close(fig_tmp)

    counts  = hb_tmp.get_array()       # (M,)
    centres = hb_tmp.get_offsets()     # (M, 2)

    # ------------------------------------------------------------------
    # 6.  SMOOTH COUNTS WITH K‑NEAREST AVERAGE
    tree             = cKDTree(centres)
    _, idxs           = tree.query(centres, k=6)       # includes self
    counts_smooth     = counts[idxs].mean(axis=1) / n_frames
    counts_raw        = counts / n_frames

    # ------------------------------------------------------------------
    # 7.  PLOT ON GLOBE
    stage = frame * 1.5 / 60 + 26                     # hpf

    fig, ax = plt.subplots(figsize=(10, 5),
                           subplot_kw=dict(projection=proj))

    hb = ax.hexbin(
            centres[:, 0], centres[:, 1],
            C=counts_smooth,
            gridsize=nbins,
            extent=extent,
            reduce_C_function=np.mean,
            cmap="inferno",
            vmin=0, vmax=1.5,
            alpha=0.8,
            linewidths=0,       # no edge lines
        edgecolors='none'
    )

    ax.set_global()                 # same as set_xlim/ylim(xlim, ylim)

    # cosmetics
    fig.colorbar(hb, ax=ax, label="cell count")
    ax.set_title(f"Deep cell density on embryonic surface ({stage:.2f} hpf)")
    ax.gridlines(draw_labels=True,
                 xformatter=LongitudeFormatter(),
                 yformatter=LatitudeFormatter())

    plt.tight_layout()

    fig.savefig(
        os.path.join(hm_path, f"deep_cell_density_f{frame:04}.png"),
        dpi=600, bbox_inches="tight"
    )
    # plt.show()

    fig.savefig(os.path.join(hm_path, f"deep_cell_density_f{frame:04}.png"),
                 dpi=600, bbox_inches="tight")