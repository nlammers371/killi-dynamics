import napari
import os
import numpy as np
import pandas as pd
import dask.array as da
import zarr
import matplotlib
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from napari.utils.colormaps import Colormap
from tqdm import tqdm
import colorcet as cc
from PIL import Image, ImageDraw, ImageFont
from napari.settings import get_settings
from napari.utils.colormaps import Colormap, AVAILABLE_COLORMAPS

settings = get_settings()
from napari.utils.colormaps import Colormap, AVAILABLE_COLORMAPS

def add_matplotlib_cmap_to_napari(name: str):
    """Convert a Matplotlib cmap to a Napari Colormap and register it."""
    if name not in matplotlib.colormaps:
        raise ValueError(f"{name} not found in Matplotlib colormaps")

    cmap = matplotlib.colormaps[name]
    rgba = cmap(np.linspace(0, 1, 256))
    napari_cmap = Colormap(rgba, name=name)

    # add to Napari registry
    AVAILABLE_COLORMAPS[name] = napari_cmap
    return name

# Example: register "Spectral"
add_matplotlib_cmap_to_napari("RdYlBu")
add_matplotlib_cmap_to_napari("RdYlGn")
add_matplotlib_cmap_to_napari("Spectral")
add_matplotlib_cmap_to_napari("PRGn")

# get napari's bop blue colormap
base = AVAILABLE_COLORMAPS["bop orange"]
rgba = np.array(base.colors)
rgba[0, -1] = 0
bop_orange = Colormap(rgba, name="bop_orange")


# get napari's bop blue colormap
base = AVAILABLE_COLORMAPS["bop blue"]

# copy its colors to numpy array
rgba = np.array(base.colors)[64:, :]

# set the first entry to white (or transparent white)
# rgba[0, :3] = [1, 1, 1]   # RGB white
rgba[0, 3] = 1.0         # alpha=1 (or 0.0 if you want transparency)
rgba[0, -1] = 0
# build a new colormap
bop_blue_white = Colormap(rgba, name="bop_blue_white")

# --- NEW: smoothing + color params ---
SMOOTH_WINDOW = 15          # in frames; odd numbers look nice with center=True
COLOR_MIN, COLOR_MAX = -1, 1   # set the color range you want for v_rad_smooth
N = 40
tail_length = 120
LCP_THRESH = 200
USE_MIP = True
MASK_R = 550
MASK_CENTER = (758.75, 791.25)  # (y, x)
HEADLESS = True
INVERT_NLS = False
DT = 60
stride = 2      # save every N frames
scale_factor = 2  # multiplier for resolution

if INVERT_NLS:
    out_suffix = "_inv"
    nls_suffix = "_r"
    text_color = "black"
    lcp_color = ("bop_blue_white", bop_blue_white)
    tracks_cmap = "RdYlGn"
else:
    out_suffix = ""
    nls_suffix = ""
    text_color = "white"
    lcp_color = ("bop_blue_white", bop_orange) #"bop orange"
    tracks_cmap = "PiYG"

if USE_MIP:
    suffix = "_mip"
else:
    suffix = ""

root = r"E:\Nick\killi_tracker"
project_name = "20241126_LCP1-NLSMSC_v0"
image_path = os.path.join(root, "built_data", "zarr_image_files",  project_name + suffix + ".zarr")
track_name = "tracking_lcp_nuclei"
tracks_csv = os.path.join(root, "tracking", project_name, track_name, "track_0000_0719", "tracks_dist.csv")
OUTDIR = Path(r"E:\Nick\killi_tracker\figures\syd_paper") / project_name / f"frames{out_suffix}"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Lazy array with time axis intact: shape like (T, Z, Y, X)
image_da = da.from_zarr(image_path)
image_z = zarr.open(image_path, mode="r")

scale_vec = (
    image_z.attrs["PhysicalSizeZ"],
    image_z.attrs["PhysicalSizeY"],
    image_z.attrs["PhysicalSizeX"],
)

if USE_MIP:
    # calculate weghted center of mass for imnage
    shape = image_da.shape
    yg, xg = np.meshgrid(
        np.arange(0, shape[2]), np.arange(0, shape[3]), indexing="ij"
    )
    im_samp = image_da[0, 1, :, :]
    im_samp = im_samp.compute().astype(float)
    # com_y = np.sum(np.multiply(yg, im_samp)) / np.sum(im_samp)
    # com_x = np.sum(np.multiply(xg, im_samp)) / np.sum(im_samp)
    dist_arr = np.sqrt((yg - MASK_CENTER[0]) ** 2 + (xg - MASK_CENTER[1]) ** 2) * scale_vec[1]
    mask = dist_arr < MASK_R
    # apply mask to channel 0
    lcp = image_da[:, 0, :, :]
    image_da[:, 0, :, :] = np.multiply(lcp, mask[None, :, :])



tracks_df = pd.read_csv(tracks_csv)

# --- NEW: compute smoothed v_radial_s per track (rolling over time) ---
tracks_df = tracks_df.sort_values(["track_id", "t"])
tracks_df["v_rad_smooth"] = (
    tracks_df.groupby("track_id")["v_radial_s"]
             .transform(lambda s: s.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean())
)
tracks_df = tracks_df.dropna(subset=["v_rad_smooth"])

if USE_MIP:
    viewer = napari.Viewer(ndisplay=2)
    scale_vec = (scale_vec[1], scale_vec[2])
else:
    viewer = napari.Viewer(ndisplay=3)

if INVERT_NLS:
    viewer.theme = "light"
else:
    viewer.theme = "dark"

# img_layer = viewer.add_image(
#     image_da[:, ::-1],
#     name="image",
#     channel_axis=1,
#     colormap=[("orange_gold", orange_gold_napari),
#               ("gray" + nls_suffix, Colormap(matplotlib.colormaps["gray" + nls_suffix](np.linspace(0, 1, 256))))][::-1],
#     scale=scale_vec,
#     contrast_limits=[(LCP_THRESH, 1500), (0, 4000)][::-1],
#     blending="translucent_no_depth",
#     opacity=0.9,
#     rgb=False,
# )
image_da = np.moveaxis(image_da, 2, 3)
ch0 = image_da[:, 0, ::-1, :]
ch1 = image_da[:, 1, ::-1, :]


layer1 = viewer.add_image(ch1, colormap="gray" + nls_suffix, opacity=0.9, blending="translucent_no_depth",
                          name="channel1", scale=scale_vec, contrast_limits=(0, 4000))

layer0 = viewer.add_image(ch0, colormap=lcp_color, opacity=0.6, blending="translucent_no_depth",  # colormap=("orange_gold", orange_gold_napari)
                          name="channel0", scale=scale_vec, contrast_limits=(LCP_THRESH, 500))


viewer.dims.set_current_step(0, int(N))  # axis 0 is time

# Tracks expect columns [track_id, t, z, y, x]
props = {
    "distance": -tracks_df["d_geo"].to_numpy(), "time": -tracks_df["t"].to_numpy(),
    "start_dist": tracks_df.groupby("track_id")["d_geo"].transform("first").to_numpy(),
    "v_rad": tracks_df["v_radial_s"].to_numpy(),
    # --- NEW: expose smoothed values to layer properties ---
    "v_rad_smooth": tracks_df["v_rad_smooth"].to_numpy(),
}

tracks_df["x_orig"] = tracks_df["x"].to_numpy()
tracks_df["y_orig"] = tracks_df["y"].to_numpy()
tracks_df["x"] = tracks_df["y_orig"].copy()
tracks_df["y"] = tracks_df["x_orig"].copy()

tracks_df["y"] = (ch0.shape[-2] - 1)*scale_vec[-2] - tracks_df["y"]

if not USE_MIP:
    fields = ["track_id", "t", "z", "y", "x"]
    scale = (1, 1, 1)
else:
    fields = ["track_id", "t", "y", "x"]
    scale = (1, 1)

tracks_layer = viewer.add_tracks(
    tracks_df[fields].to_numpy(),
    name="tracks",
    scale=scale,         # time+space scale to match image
    properties=props,
    tail_length=tail_length,
    visible=True,
    opacity=0.85,
    tail_width=3,
    blending="translucent_no_depth"
)

viewer.layers.move(viewer.layers.index("tracks"), 1)

# --- NEW: make smoothed field the default color & set color range ---
tracks_layer.color_by = "v_rad_smooth"
tracks_layer.colormap = tracks_cmap        # pick any colormap name you like
tracks_layer.color_limits = (COLOR_MIN, COLOR_MAX)

viewer.window.qt_viewer.canvas.native.setFixedSize(800, 800)
viewer.camera.center = (MASK_CENTER[0], MASK_CENTER[0])

viewer.scale_bar.visible = True
viewer.scale_bar.unit = "um"
viewer.scale_bar.color = text_color

if USE_MIP:
    # 2D: center is (y, x)
    center_world = (MASK_CENTER[0] * scale_vec[0], MASK_CENTER[1] * scale_vec[1])
else:
    # 3D: center is (z, y, x)
    center = (image_da.shape[1]/2, image_da.shape[2]/2, image_da.shape[3]/2)
    center_world = (center[0] * scale_vec[0], center[1] * scale_vec[1], center[2] * scale_vec[2])

viewer.camera.center = center_world
viewer.camera.zoom = 0.7

# --- headless snapshot export ---
if HEADLESS:
    outdir = Path(root) / "fig_snapshots"
    outdir.mkdir(parents=True, exist_ok=True)


    n_frames = image_da.shape[0]

    for i, t in enumerate(tqdm(range(0, n_frames, stride), "Exporting frames...")):
        viewer.dims.set_current_step(0, t)
        arr = viewer.screenshot(canvas_only=True, scale=scale_factor)

        font = ImageFont.truetype("arial.ttf", 24)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        t_hr = np.round((t * DT) / 3600, 1)
        label = f"{t_hr} hrs"

        # margins in pixels
        pad = 24
        # measure text if you want to right-align:
        # w, h = draw.textsize(label, font=font)
        # pos = (img.width - w - pad, pad)
        pos = (pad, pad)  # upper-left

        draw.text(pos, label, fill=text_color, font=font
                  )

        outpath = OUTDIR / f"{project_name}_iter{i:04d}.png"
        img.save(outpath)

else:
    # interactive GUI mode
    napari.run()

print("Done.")