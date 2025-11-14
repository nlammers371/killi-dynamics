import napari
import os
import numpy as np
import pandas as pd
import dask.array as da
import zarr
import matplotlib
from pathlib import Path
from tqdm import tqdm
import colorcet as cc
from PIL import Image, ImageDraw, ImageFont
from napari.utils.colormaps import Colormap, AVAILABLE_COLORMAPS, ensure_colormap
from bioio import BioImage


# get colormap
base = AVAILABLE_COLORMAPS["bop orange"]
rgba = np.array(base.colors)
rgba[0, -1] = 0
bop_orange = Colormap(rgba, name="bop_orange")
W, H = 1920, 1920
# --- NEW: smoothing + color params ---
N = 40
stride = 2      # save every N frames
scale_factor = 1  # multiplier for resolution
text_color = "white"
lcp_color = bop_orange
folder_path = Path(r"D:\Syd\MoviesForCustomLUT_Nick")
HEADLESS = True

settings_dict = {"Movie1_NLSxLCP1_Aggregation_72-96hpf": {"LCP_MIN": 125, "LCP_MAX": 300, "NLS_MIN": 100, "NLS_MAX": 2500, "DT": 90},
                 "Movie2_ALPM_NLSxLCP1_102hpfStart": {"LCP_MIN": 175, "LCP_MAX": 300, "NLS_MIN": 175, "NLS_MAX": 2500, "DT": 90},
                 "Movie3_stab-wound_Complete": {"LCP_MIN": 200, "LCP_MAX": 500, "NLS_MIN": 100, "NLS_MAX": 5000, "DT": 90},
                 "Movie4_NLSxLCP1_A830148-72_movie73-98hpf": {"LCP_MIN": 125, "LCP_MAX": 300, "NLS_MIN": 100, "NLS_MAX": 2500, "DT": 90},
                 "Movie5_LCP1_HighRes_96hpf": {"LCP_MIN": 125, "LCP_MAX": 500, "NLS_MIN": 100, "NLS_MAX": 2500, "DT": 60}}

# get list of  directories in folder path
data_folders = sorted([f.name for f in os.scandir(folder_path) if f.is_dir()])

OUTROOT= Path(r"E:\Nick\killi_immuno_paper\figures\movies")


for folder_i, folder in enumerate(tqdm(data_folders, desc="Processing folders...", position=0)):
    if folder_i < 2:
        continue
    OUTDIR = OUTROOT / folder
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # get settings for current folder
    LCP_MIN = settings_dict[folder]["LCP_MIN"]
    LCP_MAX = settings_dict[folder]["LCP_MAX"]
    NLS_MIN = settings_dict[folder]["NLS_MIN"]
    NLS_MAX = settings_dict[folder]["NLS_MAX"]
    DT = settings_dict[folder]["DT"]

    # get list of czi files in folder
    czi_list = sorted([f.path for f in os.scandir(folder_path / folder) if f.name.endswith(".czi")])
    if folder == "Movie3_stab-wound_Complete":
        root = r"E:\Nick\killi_immuno_paper"
        project_name = "20241126_LCP1-NLSMSC"
        image_path = os.path.join(root, "built_data", "zarr_image_files", project_name + "_mip.zarr")
        czi_list = [image_path]

    for czi_i, czi_path in enumerate(tqdm(czi_list, desc="Processing CZIs...", position=1, leave=False)):

        # Lazy array with time axis intact: shape like (T, Z, Y, X)
        if folder == "Movie3_stab-wound_Complete":
            image_da = da.from_zarr(czi_path)
            image_z = zarr.open(czi_path, mode="r")

            scale_vec = (
                image_z.attrs["PhysicalSizeZ"],
                image_z.attrs["PhysicalSizeY"],
                image_z.attrs["PhysicalSizeX"],
            )
        else:
            image_o = BioImage(czi_path)
            image_da = image_o.dask_data

            # get matadata
            scale_vec = np.asarray(image_o.physical_pixel_sizes)

        # calculate weghted center of mass for imnage
        shape = image_da.shape

        viewer = viewer = napari.Viewer(ndisplay=2)
        scale_vec = (scale_vec[1], scale_vec[2])
        viewer.theme = "dark"

        # permute
        if folder == "Movie3_stab-wound_Complete":
            image_da = np.moveaxis(image_da, 2, 3)
            ch0 = image_da[:, 0, ::-1, :]
            ch1 = image_da[:, 1, ::-1, :]
        else:
            ch0 = np.squeeze(image_da[:, 0, :, :])
            if folder != "Movie5_LCP1_HighRes_96hpf":
                ch1 = np.squeeze(image_da[:, 1, :, :])

        if folder != "Movie5_LCP1_HighRes_96hpf":
            layer1 = viewer.add_image(ch1, colormap="gray" , opacity=0.9, blending="translucent_no_depth",
                                      name="channel1", scale=scale_vec, contrast_limits=(NLS_MIN, NLS_MAX))

        layer0 = viewer.add_image(ch0, colormap=lcp_color, opacity=0.6, blending="translucent_no_depth",  # colormap=("orange_gold", orange_gold_napari)
                                  name="channel0", scale=scale_vec, contrast_limits=(LCP_MIN, LCP_MAX))


        viewer.dims.set_current_step(0, int(N))  # axis 0 is time

        viewer.window.qt_viewer.canvas.native.setFixedSize(H, W)
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "um"
        viewer.scale_bar.color = text_color
        viewer.camera.zoom = 0.7
        if folder == "Movie2_ALPM_NLSxLCP1_102hpfStart":
            layer0.rotate = 60
            layer1.rotate = 60
            dx = -250
            dy = 850
            layer0.translate = (dy, dx)
            layer1.translate = (dy, dx)

        # --- headless snapshot export ---
        if HEADLESS:
            outdir = Path(OUTDIR) / f"fig_snapshots_side{czi_i:02d}"
            outdir.mkdir(parents=True, exist_ok=True)

            n_frames = image_da.shape[0]

            for i, t in enumerate(tqdm(range(0, n_frames, stride), desc="Exporting frames...", position=2, leave=False)):
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

                draw.text(pos, label, fill=text_color, font=font)

                outpath = outdir / f"{folder}_side{czi_i:02d}_iter{i:04d}.png"
                img.save(outpath)

        else:
            # interactive GUI mode
            napari.run()

print("Done.")