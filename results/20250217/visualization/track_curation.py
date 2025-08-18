import os
import numpy as np
import pandas as pd
import dask.array as da
import napari
from pathlib import Path
from typing import Union
from magicgui import magicgui
from napari.utils.notifications import show_info

def get_nearby_tracks(df, track_id, z_var, dist_um=90):
    main = df[df["track_id"] == int(track_id)][["t", z_var, "y", "x"]].rename(
        columns={z_var: "z_ref", "y": "y_ref", "x": "x_ref"}
    )
    merged = df.merge(main, on="t")
    merged["dist"] = np.sqrt(
        (merged["y"] - merged["y_ref"])**2 +
        (merged["x"] - merged["x_ref"])**2 +
        (merged[z_var] - merged["z_ref"])**2
    )
    prox_ids = merged.loc[
        (merged["dist"] < dist_um) & (merged["track_id"] != int(track_id)), "track_id"
    ].unique()
    return df[df["track_id"].isin(prox_ids)]

def get_single_track_df(tracks_df, track_id, z_var="z_mip"):
    if track_id is None or track_id == '':
        return None
    mask = tracks_df["track_id"].astype(int) == int(track_id)
    if mask.sum() == 0:
        return None
    if z_var == "z_mip":
        return tracks_df.loc[mask, ["track_id", "t", z_var, "z", "y", "x"]]
    else:
        return tracks_df.loc[mask, ["track_id", "t", "z", "y", "x"]]

def set_qt_env():
    os.environ["QT_API"] = "pyqt5"
    os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"

def load_tracks_df(csv_path, div_z=None, mip_flag=False):
    df = pd.read_csv(csv_path)
    if mip_flag and div_z is not None and "z" in df.columns:
        df["z_mip"] = np.floor(df["z"] / div_z)
    return df

def main(root: Union[str, Path], project_name: str, mip_flag: bool = True):
    set_qt_env()

    root = Path(root)
    zpath = root / "built_data" / "zarr_image_files" / f"{project_name}_fused.zarr"
    mip_path = root / "built_data" / "zarr_image_files" / f"{project_name}_mip.zarr"

    fused_image_zarr = da.from_zarr(zpath)
    mip_zarr = da.from_zarr(mip_path)
    div_z = fused_image_zarr.shape[2] // 2

    print(f"Loading NLS tracking data for project: {project_name}")
    nls_track_path = os.path.join(root, "tracking", project_name, "tracking_20250328_redux", "well0000", "track_0000_2339_cb")
    nls_tracks_csv = os.path.join(nls_track_path, "tracks_fluo.csv")
    nls_tracks_df = load_tracks_df(nls_tracks_csv, div_z, mip_flag)
    z_var = "z_mip" if mip_flag else "z"

    print(f"Loading LCP tracking data for project: {project_name}")
    lcp_track_path = os.path.join(root, "built_data", "tracking", project_name)
    lcp_tracks_csv = os.path.join(lcp_track_path, "lcp_tracks_df.csv")
    lcp_tracks_df = load_tracks_df(lcp_tracks_csv, div_z, mip_flag)

    if project_name == "20250311_LCP1-NLSMSC":
        lcp_tracks_df["z"] = lcp_tracks_df["z"] / 3
        lcp_tracks_df["z_mip"] = np.floor(lcp_tracks_df["z"] / div_z)

    print("Initializing napari...")
    viewer = napari.Viewer()

    # Default mode: MIP
    scale_vec = (1, 1, 1)
    viewer.add_image(
        mip_zarr, channel_axis=1, scale=scale_vec, name="main_image",
        colormap=["cyan", "gray"], visible=True,
        contrast_limits=[(50, 300), (0, 5000)]
    )

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    # State for track highlighting across widgets
    current_highlight = {"layer_name": None, "track_id": None, "dist_lim_um": None}

    # For toggling mode and updating image/tracks
    @magicgui(
        mode={"choices": ["3D", "MIP"], "label": "Mode"},
        call_button="Apply"
    )
    def mode_dim_widget(mode="MIP"):
        # Remove all previous image layers named "main_image"
        for lyr in list(viewer.layers):
            if lyr.name.startswith("main_image"):
                viewer.layers.remove(lyr)
        # Remove any selected/nearby tracks
        for lyr in list(viewer.layers):
            if lyr.name in ("selected track", "nearby tracks"):
                viewer.layers.remove(lyr)
        # Add new image layer with appropriate scale/colormap
        scale_vec = (1, 1, 1) if mode == "MIP" else (3, 1, 1)
        if mode == "3D":
            viewer.add_image(
                fused_image_zarr, channel_axis=1, scale=scale_vec, name="main_image",
                colormap=["cyan", "gray"], visible=True,
                contrast_limits=[(50, 300), (0, 5000)]
            )
            mode_dim_widget.z_var = "z"
            mode_dim_widget.scale_vec = scale_vec
            viewer.dims.ndisplay = 3
        else:
            viewer.add_image(
                mip_zarr, channel_axis=1, scale=scale_vec, name="main_image",
                colormap=["cyan", "gray"], visible=True,
                contrast_limits=[(50, 300), (0, 5000)]
            )
            viewer.dims.ndisplay =2
            mode_dim_widget.z_var = "z_mip"
            mode_dim_widget.scale_vec = scale_vec

        # -- If a track was highlighted, refresh it in the new mode --
        if (
            current_highlight["layer_name"] is not None
            and current_highlight["track_id"] not in (None, "")
        ):
            highlight_track_widget(
                current_highlight["layer_name"],
                current_highlight["track_id"],
                current_highlight["dist_lim_um"]
            )

    # Set attributes so we can access them elsewhere
    mode_dim_widget.z_var = z_var
    mode_dim_widget.scale_vec = scale_vec

    viewer.window.add_dock_widget(mode_dim_widget, area="right")

    # The available track dataframes
    track_dfs = {
        "nls tracks": nls_tracks_df,
        "lcp tracks": lcp_tracks_df,
    }

    @magicgui(
        call_button="Highlight Track",
        layer_name={"choices": list(track_dfs.keys()), "label": "Track Layer"},
        track_id={"label": "Track ID"},
        dist_lim_um={"label": "Nearby distance", "min": 1, "max": 200, "step": 1, "widget_type": "SpinBox"},
    )
    def highlight_track_widget(layer_name, track_id: str, dist_lim_um: int = 90):
        # Remove previous single-track and nearby layers
        for lyr in list(viewer.layers):
            if lyr.name in ("selected track", "nearby tracks"):
                viewer.layers.remove(lyr)
        # Use current z_var/scale for the mode
        z_var = mode_dim_widget.z_var
        scale_vec = mode_dim_widget.scale_vec
        df = get_single_track_df(track_dfs[layer_name], track_id, z_var=z_var)
        if df is None:
            print(f"Track ID {track_id} not found in {layer_name}")
            return
        first_row = df.iloc[0]
        t = int(first_row["t"])
        z = int(first_row[z_var])
        viewer.dims.set_current_step(0, t)
        viewer.dims.set_current_step(1, z)
        show_info(f"Moved viewer to t={t}, z={z}")
        # Add single selected track
        viewer.add_tracks(
            df[["track_id", "t", z_var, "y", "x"]],
            name="selected track",
            scale=scale_vec,
            visible=True,
            tail_width=8,
            tail_length=40,
            opacity=1.0,
        )
        # Add nearby tracks
        nearby = get_nearby_tracks(track_dfs[layer_name], track_id, z_var, dist_lim_um)
        if not nearby.empty:
            viewer.add_tracks(
                nearby[["track_id", "t", z_var, "y", "x"]],
                name="nearby tracks",
                scale=scale_vec,
                visible=True,
                tail_width=2,
                tail_length=40,
                opacity=0.5,
                features=nearby[["track_id"]],
            )
        # Store the highlight state for use by the mode widget
        current_highlight["layer_name"] = layer_name
        current_highlight["track_id"] = track_id
        current_highlight["dist_lim_um"] = dist_lim_um

    viewer.window.add_dock_widget(highlight_track_widget, area='right')
    napari.run()

if __name__ == "__main__":
    root = r"E:\Nick\Cole Trapnell's Lab Dropbox\Nick Lammers\Nick\killi_tracker"
    project_name = "20250311_LCP1-NLSMSC"
    mip_flag = True
    main(root=root, project_name=project_name, mip_flag=mip_flag)
