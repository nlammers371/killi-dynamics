import os
import numpy as np
import pandas as pd
import dask.array as da
import napari
from pathlib import Path
from typing import Union
from magicgui import magicgui
from napari.utils.notifications import show_info

def get_single_track_df(tracks_df, track_id, z_var="z_mip"):
    mask = tracks_df["track_id"].astype(int) == int(track_id)
    if mask.sum() == 0:
        return None
    return tracks_df.loc[mask, ["track_id", "t", z_var, "y", "x"]]

def set_qt_env():
    os.environ["QT_API"] = "pyqt5"
    os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"

def load_tracks_df(csv_path, div_z=None, mip_flag=False):
    df = pd.read_csv(csv_path)
    if mip_flag and div_z is not None and "z" in df.columns:
        df["z_mip"] = np.floor(df["z"] / div_z)
    return df

def main(root: Union[str, Path],
         project_name: str,
         mip_flag: bool = True,):
    set_qt_env()

    # side = 0  # for multi-FOV images, not currently used

    # Image paths
    root = Path(root)
    zpath = root / "built_data" / "zarr_image_files"/ f"{project_name}_fused.zarr"
    mip_path = root / "built_data" / "zarr_image_files" / f"{project_name}_mip.zarr"

    fused_image_zarr = da.from_zarr(zpath)
    mip_zarr = da.from_zarr(mip_path)
    div_z = fused_image_zarr.shape[2] // 2

    # NLS tracks
    print(f"Loading NLS tracking data for project: {project_name}")
    nls_track_path = os.path.join(root, "tracking", project_name, "tracking_20250328_redux", "well0000", "track_0000_2339_cb")
    nls_tracks_csv = os.path.join(nls_track_path, "tracks_fluo.csv")
    nls_tracks_df = load_tracks_df(nls_tracks_csv, div_z, mip_flag)
    z_var = "z_mip" if mip_flag else "z"

    # LCP tracks
    print(f"Loading LCP tracking data for project: {project_name}")
    lcp_track_path = os.path.join(root, "built_data", "tracking", project_name)
    lcp_tracks_csv = os.path.join(lcp_track_path, "lcp_tracks_df.csv")
    lcp_tracks_df = load_tracks_df(lcp_tracks_csv, div_z, mip_flag)
    if project_name == "20250311_LCP1-NLSMSC":  # patch  fix for now
        lcp_tracks_df["z"] = lcp_tracks_df["z"] / 3
        lcp_tracks_df["z_mip"] = np.floor(lcp_tracks_df["z"] / div_z)

    # Image scaling
    scale_vec = (1, 1, 1) if mip_flag else (3, 1, 1)

    print("Initializing napari...")
    viewer = napari.Viewer()

    if not mip_flag:
        image_layer = viewer.add_image(
            fused_image_zarr, channel_axis=1, scale=scale_vec,
            colormap=["cyan", "gray"], visible=False,
            contrast_limits=[(50, 300), (0, 2500)]
        )
    else:
        image_layer = viewer.add_image(
            mip_zarr, channel_axis=1, scale=scale_vec,
            colormap=["cyan", "gray"], visible=False,
            contrast_limits=[(50, 300), (0, 2500)]
        )

    # Add both tracks, store napari layers
    nls_tracks_layer = viewer.add_tracks(
        nls_tracks_df[["track_id", "t", z_var, "y", "x"]],
        name="nls tracks",
        scale=scale_vec,
        visible=True,
        tail_width=1,
        tail_length=40,
        opacity=0.3,
        features=nls_tracks_df[["track_id"]],
    )

    lcp_tracks_layer = viewer.add_tracks(
        lcp_tracks_df[["track_id", "t", z_var, "y", "x"]],
        name="lcp tracks (bright)",
        scale=scale_vec,
        visible=True,
        tail_width=1,
        tail_length=40,
        opacity=0.3,
        features=lcp_tracks_df[["track_id"]],
    )

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    # ---- MAGICGUI WIDGET ----
    layers_dict = {
        "nls tracks": nls_tracks_layer,
        "lcp tracks (bright)": lcp_tracks_layer,
    }
    track_dfs = {
        "nls tracks": nls_tracks_df,
        "lcp tracks (bright)": lcp_tracks_df,
    }

    # For cleanup: keep reference to selected track layer

    selected_layer_holder = {"layer": None}

    @magicgui(
        call_button="Highlight Track",
        layer_name={"choices": list(layers_dict.keys()), "label": "Track Layer"},
        track_id={"label": "Track ID"},
    )
    def highlight_track_widget(layer_name, track_id: str):
        # Remove old selected layer if it exists
        if selected_layer_holder["layer"] is not None:
            viewer.layers.remove(selected_layer_holder["layer"])
            selected_layer_holder["layer"] = None

        df = get_single_track_df(track_dfs[layer_name], track_id, z_var=z_var)
        if df is None:
            print(f"Track ID {track_id} not found in {layer_name}")
            return

        # Move napari sliders to the first time/z of the track
        first_row = df.iloc[0]
        t = int(first_row["t"])
        z = int(first_row[z_var])
        viewer.dims.set_current_step(0, t)
        viewer.dims.set_current_step(1, z)
        show_info(f"Moved viewer to t={t}, z={z}")

        # Add bold, labeled layer for just this track
        layer = viewer.add_tracks(
            df[["track_id", "t", z_var, "y", "x"]],
            name=f"selected track {track_id}",
            scale=scale_vec,
            visible=True,
            tail_width=8,
            tail_length=40,
            opacity=1.0,
            # color="yellow",        <-- REMOVE THIS LINE!
            # text={"string": "{track_id}", "size": 18, "color": "yellow", "visible": True, "anchor": "head"},
        )

        layer.show_id = True
        # show_info(f"Togged ID display to {layer.show_id}")
        selected_layer_holder["layer"] = layer

    viewer.window.add_dock_widget(highlight_track_widget, area='right')

    napari.run()

if __name__ == "__main__":
    root = r"E:\Nick\Cole Trapnell's Lab Dropbox\Nick Lammers\Nick\killi_tracker"
    project_name = "20250311_LCP1-NLSMSC"
    mip_flag = True
    main(root=root, project_name=project_name, mip_flag=mip_flag)
