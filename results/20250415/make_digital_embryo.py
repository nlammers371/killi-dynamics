import zarr
import napari
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm
import re
import pandas as pd
from src.geometry import create_sphere_mesh
from qtpy.QtWidgets import QApplication
from napari_animation import Animation

if __name__ == "__main__":

    os.environ["QT_API"] = "pyqt5"
    overwrite = False
    # get filepaths
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project = "20250311_LCP1-NLSMSC"
    stitch_suffix = ""

    # make output path
    output_dir = os.path.join(root, "figures", project, "digital_embryo", "")
    os.makedirs(output_dir, exist_ok=True)

    # set params
    start_angle = 220
    initial_angle = (-3.8195498904405216, 8.89243987179023 + start_angle, 105.88077856425232)
    zoom = 0.58
    n_revs = 3.5
    frame_increment = 2
    scale_vec = tuple([3, 1, 1])
    stop_angle = 40
    final_angle = n_revs * 360 + stop_angle

    nls_track_path = os.path.join(root, "tracking", project, "tracking_20250328_redux", "well0000",
                                  "track_0000_2339_cb", "")
    nls_track_zarr = zarr.open(os.path.join(nls_track_path, "segments.zarr"), mode="r")
    nls_tracks_df = pd.read_csv(os.path.join(nls_track_path, "tracks_fluo" + stitch_suffix + ".csv"))
    sphere_df = pd.read_csv(os.path.join(nls_track_path, "sphere_fit.csv"))

    # add class info to tracks
    nucleus_class_df = pd.read_csv(os.path.join(nls_track_path, "track_class_df_full.csv"))
    nls_tracks_df = nls_tracks_df.merge(nucleus_class_df.loc[:, ["track_id", "t", "track_class", "frame_class"]],
                                        on=["track_id", "t"], how="left")

    cell_type_filter_vec = [np.asarray([0])] #[np.asarray([0, 1, 2]), np.asarray([0]), np.asarray([1]), np.asarray([2])] #
    cell_type_names = ["deep2"] #["all2", "deep2", "evl2", "yolk2"] #["all"]
    colormap_vec = ["I Purple"] #["turbo", "I Purple", "I Forest", "red"] #["turbo"]

    window = 50
    sm = (
        sphere_df[["xs", "ys", "zs"]]
        .rolling(window, center=True, min_periods=1)
        .mean()
        .rename(columns={"xs": "xsm", "ys": "ysm", "zs": "zsm"})
    )
    sphere_df[["xsm", "ysm", "zsm"]] = sm

    # center tracks
    nls_tracks_df["z_scaled"] = nls_tracks_df["z"].copy() * 3
    nls_tracks_df = nls_tracks_df.merge(sphere_df.loc[:, ["t" ,"xsm", "ysm", "zsm"]], on=["t"], how="left")
    nls_tracks_df[["xs", "ys", "zs"]] = nls_tracks_df[["x", "y", "z_scaled"]] - nls_tracks_df[["xsm", "ysm", "zsm"]].to_numpy()

    sphere_center = np.asarray([0, 0, 0])
    sphere_radius = 0.9 * sphere_df.loc[sphere_df["t"] == 1500, "r"].to_numpy()[0]
    sphere_mesh = create_sphere_mesh(sphere_center, sphere_radius, 100)
    num_frames = sphere_df.shape[0]

    angle_vec = -np.linspace(0, final_angle, num_frames)

    frames_to_process = np.arange(0, num_frames, frame_increment)

    for i, cell_type in enumerate(tqdm(cell_type_names, desc="Processing cell types", unit="cell type")):

        cmap = colormap_vec[i]
        cell_ids = cell_type_filter_vec[i]
        tracks_to_plot = nls_tracks_df.loc[nls_tracks_df["track_class"].isin(cell_ids), :]
        name = cell_type_names[i]

        iter_dir = os.path.join(root, "figures", project, "digital_embryo", name, "")
        os.makedirs(iter_dir, exist_ok=True)
        # initialize viewer
        viewer = napari.Viewer(ndisplay=3)

        # set viewer options
        # Enable the built-in scale bar
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = 'µm'
        viewer.scale_bar.font_size = 18

        viewer.camera.angles = tuple([initial_angle[0], initial_angle[1], initial_angle[2]])
        viewer.camera.zoom = zoom

        # initialize mesh
        mesh = viewer.add_surface(sphere_mesh, name="sphere mesh", shading='smooth')
        # mesh.wireframe.visible = True
        # mesh.wireframe.color = (1, 1, 1, 1)  # RGBA white
        mesh.opacity = 0.6  # faces become fully transparent
        bob_blue_rgba = (30 / 255, 144 / 255, 255 / 255, 0.4)  # DodgerBlue, 40 % opaque

        mesh.vertex_colors = np.tile(bob_blue_rgba, (len(sphere_mesh[0]), 1))

        tl = 50
        if i >= 2:
            tl = 75
        # add tracks
        viewer.add_tracks(
            tracks_to_plot[["track_id", "t", "zs", "ys", "xs"]],
            name="nls tracks",
            blending="translucent",
            colormap=cmap,
            # scale=tuple(scale_vec),
            translate=(0, 0, 0, 0),
            # features=nls_tracks_df[["fluo_mean", "nucleus_volume"]],
            visible=True,
            tail_width=3,
            tail_length=tl
        )

        # anim = Animation(viewer)
        frames = []
        for ref_frame in tqdm(frames_to_process, desc="Processing frames", unit="frame"):

            viewer.camera.angles = tuple([initial_angle[0], initial_angle[1] + angle_vec[ref_frame], initial_angle[2]])

            # viewer.dims.current_step = (ref_frame % nls_tracks_df.loc[:, "t"].to_numpy().max(),)
            viewer.dims.set_current_step(0, ref_frame)

            QApplication.processEvents()

            screenshot_path = os.path.join(iter_dir, name + f"_frame_{ref_frame:04}.png")
            frame = viewer.window.screenshot(canvas_only=True, path=screenshot_path)

        # frames.append(frame)

        # save screenshot#

        # screenshot.save(screenshot_path)
        # anim.capture_keyframe(steps=1)

    # 2a.  write a video directly (requires ffmpeg in PATH)
    # iio.imwrite(
    #    os.path.join(output_dir, "mesh_tracks_spin2.mp4"),
    #     np.stack(frames),  # (t, h, w, 4)
    #     fps=30,
    #     codec="libx264",  # default
    #     bitrate="10M"  # bump if the video looks blocky
    # )


    print("Check")

