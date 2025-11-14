import zarr
import napari
import numpy as np
import pandas as pd
import os
import dask.array as da


if __name__ == "__main__":

    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    os.environ["QT_API"] = "pyqt5"
    os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
    os.environ["QT_API"] = "pyqt5"

    project_name = "20250311_LCP1-NLSMSC"
    mip_flag = False  # assume along z axis for now
    stitch_suffix = ""
    side = 0

    # load imaga dataset
    zpath = os.path.join(root, "built_data", "zarr_image_files", project_name + "_fused.zarr")
    mip_path = os.path.join(root, "built_data", "zarr_image_files", project_name + "_mip.zarr")
    fused_image_zarr = da.from_zarr(zpath)
    mip_zarr = da.from_zarr(mip_path)#[:, :, side, :, :]

    div_z = fused_image_zarr.shape[2] // 2

    # load full tracking dataset
    print("Loading tracking data for project:", project_name)
    nls_track_path = os.path.join(root, "tracking", project_name, "tracking_20250328_redux", "well0000", "track_0000_2339_cb", "")
    nls_track_zarr = zarr.open(os.path.join(nls_track_path, "segments.zarr"), mode="r")
    nls_tracks_df = pd.read_csv(os.path.join(nls_track_path, "tracks_fluo" + stitch_suffix + ".csv"))
    nucleus_class_df = pd.read_csv(os.path.join(nls_track_path, "track_class_df_full.csv"))

    # add class info to tracks
    nls_tracks_df = nls_tracks_df.merge(nucleus_class_df.loc[:, ["track_id", "t", "track_class", "frame_class"]], on=["track_id", "t"], how="left")

    z_var = "z"
    if mip_flag:
        z_var = "z_mip"
        nls_tracks_df["z_mip"] = np.floor(nls_tracks_df["z"]/div_z)
        # nls_tracks_df = nls_tracks_df.loc[nls_tracks_df["z_mip"] == side]

    # load nucleus class dataset

    # get scale info
    if mip_flag:
        scale_vec = tuple([1, 1, 1])
    else:
        scale_vec = tuple([3, 1, 1])

    viewer = napari.Viewer()
    if not mip_flag:
        viewer.add_image(fused_image_zarr, channel_axis=1, scale=scale_vec, colormap=["cyan", "gray"], visible=False, contrast_limits=[(0, 500), (0, 2500)])
    else:
        viewer.add_image(mip_zarr, channel_axis=1, scale=scale_vec, colormap=["cyan", "gray"], visible=False, contrast_limits=[(0, 500), (0, 2500)])

    viewer.add_tracks(
        nls_tracks_df[["track_id", "t", z_var, "y", "x"]],
        name="nls tracks",
        scale=tuple(scale_vec),
        color_by="track_class",
        properties=nls_tracks_df[["track_class", "frame_class"]],
        translate=(0, 0, 0, 0),
        # features=nls_tracks_df[["fluo_mean", "nucleus_volume"]],
        visible=False,
        tail_width=3,
        tail_length=40
    )

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    napari.run()

    print("check")
