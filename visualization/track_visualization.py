import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import dask.array as da
import zarr
import napari
from magicgui import magicgui
from napari.utils.notifications import show_info, show_warning

# ROI point-in-polygon
try:
    from matplotlib.path import Path as MplPath
except Exception as e:
    MplPath = None
    print("[ROI] matplotlib is required for ROI selection:", e)


# ----------------------------- Config -----------------------------

PRIMARY_TRACK_SET = "nls tracks"  # or "lcp tracks"

P_LOW = 1.0
P_HIGH = 99.99
# ---- Contrast estimation controls ----
USE_CONTRAST_SAMPLING = True      # set False to skip computing limits entirely
ENDPOINTS_ONLY = True             # if True, sample only first & last time frames

USE_JITTER = True
USE_RANDOM_VOXELS = False
RANDOM_VOXELS_COUNT = 3_000_000

ALL_THIN_T_STEP = 1
ALL_TAIL_LENGTH = 30
ALL_TAIL_WIDTH = 2


# ----------------------------- Utilities -----------------------------
def _arr_for_contrast(arr: da.core.Array) -> da.core.Array:
    """Return a small subset for contrast estimation."""
    if not USE_CONTRAST_SAMPLING:
        return arr  # won't be used, caller will skip
    if not ENDPOINTS_ONLY:
        return arr  # full array (robust sampler will sub-sample internally)

    # endpoints-only: use only frame 0 and last frame (axis 0 = time)
    try:
        T = int(arr.shape[0]) if arr.shape[0] is not None else 0
    except Exception:
        T = 0
    if T <= 1:
        return arr[0:1]           # single frame case
    # concat is lazy in dask, still cheap
    return da.concatenate([arr[0:1], arr[T-1:T]], axis=0)

def set_qt_env():
    os.environ["QT_API"] = "pyqt5"
    os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"


def safe_from_zarr(zpath: Union[str, Path]) -> Optional[da.core.Array]:
    try:
        return da.from_zarr(str(zpath))
    except Exception:
        return None


def read_channel_names(zpath: Union[str, Path]) -> Optional[List[str]]:
    root = zarr.open(str(zpath), mode="r")
    return list(root.attrs["Channels"])


def default_colormap_for_channel(name: str, idx: int) -> str:
    if name and ("nls" in name.lower()):
        return "gray"
    palette = ["cyan", "magenta", "yellow", "green", "red", "blue", "orange"]
    return palette[idx % len(palette)]


def _random_voxel_indices(shape, n_samples, rng):
    size = int(np.prod([int(s) for s in shape]))
    n = int(min(n_samples, size))
    if n < size:
        idx = rng.choice(size, size=n, replace=False)
    else:
        idx = np.arange(size)
    return np.unravel_index(idx, shape)


def robust_contrast_limits_per_channel(
    darr: da.core.Array,
    multichannel: bool = True,
    p_low: float = P_LOW,
    p_high: float = P_HIGH,
    random_voxels_count: int = RANDOM_VOXELS_COUNT,
) -> List[Tuple[float, float]]:

    if not multichannel:
        darr = darr[:, None, :, :, :]

    n_channels = darr.shape[1]
    limits: List[Tuple[float, float]] = []
    rng = np.random.default_rng()

    for c in range(n_channels):
        # compute the full channel slice into memory (numpy)
        chan = darr[:, c].compute()

        # draw random linear indices over the numpy array (fast)
        size = chan.size
        if size == 0:
            limits.append((0.0, 1.0))
            continue
        take = min(random_voxels_count, size)
        ridx = rng.choice(size, size=take, replace=False)
        flat = chan.ravel()[ridx]

        # robust percentiles on numpy sample
        sample = np.asarray(flat)
        lo_v = np.percentile(sample, p_low) if sample.size else 0.0
        hi_v = np.percentile(sample, p_high) if sample.size else 1.0

        if not np.isfinite(lo_v) or not np.isfinite(hi_v) or lo_v == hi_v:
            try:
                mx = np.nanmax(sample) if sample.size else 1.0
                lo_v, hi_v = 0.0, float(mx) if np.isfinite(mx) and mx > 0 else 1.0
            except Exception:
                lo_v, hi_v = 0.0, 1.0

        limits.append((float(lo_v), float(hi_v)))

    return limits


def load_tracks_df(csv_path: Union[str, Path], div_z: Optional[int] = None, mip_flag: bool = False) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        show_warning(f"Could not read tracks CSV at {csv_path}: {e}")
        return None
    if mip_flag and div_z is not None and "z" in df.columns:
        df["z_mip"] = np.floor(df["z"] / div_z).astype(int)
    return df


def infer_color_features(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None:
        return []
    candidates = [
        "mean_fluo",
        "speed",
        "size",
        "fluo_trend",
        "speed_trend",
        "intensity",
        "cluster_id",
        "track_id",
        "t",
    ]
    return [c for c in candidates if c in df.columns]


# ----------------- NEW: single-source-of-truth for z key -----------------
def choose_z_key(df: pd.DataFrame, current_mode: str, mip_zarr: Optional[da.core.Array]) -> Optional[str]:
    """
    - 3D mode: use real z if present.
    - MIP mode:
        * multi-side MIP stored as TCZYX (ndim >= 5): use z_mip if present
        * single-side MIP stored as TCYX (ndim == 4): no z column (return None)
    """
    if current_mode == "3D" and "z" in df.columns:
        return "z"
    # MIP case
    if mip_zarr is not None and getattr(mip_zarr, "ndim", 0) >= 5:  # TCZYX → multi-side MIP
        return "z_mip" if "z_mip" in df.columns else None
    # Single-side MIP (TCYX) → no z in tracks
    return None



def get_nearby_tracks(df: pd.DataFrame, track_id, z_var: Optional[str], dist_um=90):
    base_cols = ["t", "y", "x"]
    if z_var:
        base_cols = ["t", z_var, "y", "x"]

    main = df[df["track_id"] == int(track_id)][base_cols].rename(
        columns={(z_var if z_var else "t"): ("z_ref" if z_var else "t"),
                 "y": "y_ref", "x": "x_ref"}
    )
    # If z_var is None we won't use the renamed z_ref anyway
    if z_var:
        main = main.rename(columns={z_var: "z_ref"})

    if main.empty:
        return df.iloc[0:0]
    merged = df.merge(main, on="t")

    if z_var:
        merged["dist"] = np.sqrt(
            (merged["y"] - merged["y_ref"]) ** 2
            + (merged["x"] - merged["x_ref"]) ** 2
            + (merged[z_var] - merged["z_ref"]) ** 2
        )
    else:
        merged["dist"] = np.sqrt(
            (merged["y"] - merged["y_ref"]) ** 2
            + (merged["x"] - merged["x_ref"]) ** 2
        )

    prox_ids = merged.loc[
        (merged["dist"] < dist_um) & (merged["track_id"] != int(track_id)), "track_id"
    ].unique()
    return df[df["track_id"].isin(prox_ids)]


def get_single_track_df(tracks_df: pd.DataFrame, track_id, z_var: Optional[str] = "z_mip"):
    if track_id is None or track_id == "":
        return None
    mask = tracks_df["track_id"].astype(int) == int(track_id)
    if mask.sum() == 0:
        return None
    if z_var is None:
        cols = ["track_id", "t", "y", "x"]
    elif z_var == "z_mip" and "z_mip" in tracks_df.columns:
        cols = ["track_id", "t", "z_mip", "z", "y", "x"]
    else:
        cols = ["track_id", "t", "z", "y", "x"]
    return tracks_df.loc[mask, cols].copy()


def add_tracks_with_coloring(
    viewer: napari.Viewer,
    df: pd.DataFrame,
    z_key: Optional[str],
    name: str,
    scale: Tuple[float, float, float],
    tail_width: int = 4,
    tail_length: int = 40,
    opacity: float = 1.0,
    color_by: Optional[str] = None,
    colormap: str = "viridis",
):
    if z_key:
        data = df[["track_id", "t", z_key, "y", "x"]].to_numpy().astype("float32", copy=False)
    else:
        data = df[["track_id", "t", "y", "x"]].to_numpy().astype("float32", copy=False)
    properties = df.copy()

    kwargs = {}
    if color_by and (color_by in properties.columns):
        kwargs["color_by"] = color_by
        kwargs["colormap"] = colormap

    layer = viewer.add_tracks(
        data,
        name=name,
        scale=scale,
        tail_width=tail_width,
        tail_length=tail_length,
        opacity=opacity,
        properties=properties,
        **kwargs,
    )
    try:
        layer.blending = "translucent_no_depth"
    except Exception:
        pass
    return layer


def add_all_tracks_layer(
    viewer,
    track_dfs: Dict[str, pd.DataFrame],
    current_mode: str,
    mip_zarr: Optional[da.core.Array],
    primary_set=PRIMARY_TRACK_SET,
    thin_t_step=ALL_THIN_T_STEP,
    z_fallback="z",
    layer_name="All tracks",
    scale=(1, 1, 1),
    tail_len=ALL_TAIL_LENGTH,
    tail_w=ALL_TAIL_WIDTH,
    color_by_default="track_id",
    colormap_default="viridis",
):
    df = track_dfs.get(primary_set)
    if df is None or df.empty:
        return None

    if thin_t_step and thin_t_step > 1 and "t" in df.columns:
        df = df[df["t"] % int(thin_t_step) == 0].copy()

    # single source of truth, now aware of mip_zarr shape
    z_key = choose_z_key(df, current_mode, mip_zarr)

    if layer_name in viewer.layers:
        viewer.layers.remove(layer_name)

    lyr = add_tracks_with_coloring(
        viewer=viewer,
        df=df,
        z_key=z_key,
        name=layer_name,
        scale=scale,
        tail_width=tail_w,
        tail_length=tail_len,
        opacity=0.9,
        color_by=color_by_default,
        colormap=colormap_default,
    )
    return lyr


# ------------------------------ Main App ------------------------------

def main(root: Union[str, Path],
         project_name: str,
         tracking_config: str = "tracking_20250328_redux",
         tracking_instance: str = "",
         mip_flag: bool = True):
    set_qt_env()

    root = Path(root)
    zpath = root / "built_data" / "zarr_image_files" / f"{project_name}_fused.zarr"
    if not zpath.exists():
        zpath = root / "built_data" / "zarr_image_files" / f"{project_name}.zarr"
    mip_path = root / "built_data" / "zarr_image_files" / f"{project_name}_mip.zarr"

    image_zarr = safe_from_zarr(zpath)
    mip_zarr = safe_from_zarr(mip_path)

    # tracking instance inference
    if tracking_instance == "" and image_zarr is not None:
        tracking_instance = f"track_0000_{int(image_zarr.shape[0]):04}"

    # div_z heuristic for z_mip
    div_z = None
    if image_zarr is not None and image_zarr.ndim >= 3:
        div_z = int(max(int(image_zarr.shape[2]) // 2, 1))

    # --------- Load tracks ----------
    print(f"Loading tracking data for project: {project_name}")
    track_dfs: Dict[str, Optional[pd.DataFrame]] = {}

    track_path = root / "tracking" / project_name / tracking_config / tracking_instance
    tracks_csv = track_path / "tracks_fluo_stitched.csv"
    if not tracks_csv.exists():
        tracks_csv = track_path / "tracks_stitched.csv"
    if not tracks_csv.exists():
        tracks_csv = track_path / "tracks_fluo.csv"
    if not tracks_csv.exists():
        tracks_csv = track_path / "tracks.csv"


    tracks_df = load_tracks_df(tracks_csv, div_z=div_z, mip_flag=mip_flag)
    if "track_mostly_stationary" in tracks_df.columns:
        tracks_df = tracks_df[tracks_df["track_mostly_stationary"] == False].copy()
    if tracks_df is not None:
        track_dfs["nls tracks"] = tracks_df

    if not track_dfs:
        raise FileNotFoundError("No track CSVs found; please check your paths.")

    # -------------- Viewer --------------
    viewer = napari.Viewer()
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    # --------- Color-by control ---------
    all_colorables: List[str] = []
    for df in track_dfs.values():
        all_colorables.extend(infer_color_features(df))
    seen = set()
    colorables = [c for c in all_colorables if not (c in seen or seen.add(c))]
    default_color_by = "mean_fluo" if "mean_fluo" in colorables else (colorables[0] if colorables else "track_id")

    @magicgui(
        call_button="Apply Colors",
        color_by={"choices": colorables if colorables else ["track_id"], "label": "Color tracks by"},
        colormap={"choices": ["viridis", "magma", "plasma", "inferno", "turbo", "gray"], "label": "Track colormap"},
    )
    def color_widget(color_by: str = default_color_by, colormap: str = "viridis"):
        for lyr_name in viewer.layers:
            layer = viewer.layers[lyr_name]
            if hasattr(layer, "properties") and hasattr(layer, "color_by"):
                if color_by in getattr(layer, "properties", {}):
                    layer.color_by = color_by
                    layer.colormap = colormap
                else:
                    if hasattr(layer, "properties") and "track_id" in layer.properties:
                        layer.color_by = "track_id"
                        layer.colormap = colormap

    viewer.window.add_dock_widget(color_widget, area="right")

    # --------- Start with ONE image (default MIP), and matching tracks ---------
    current_mode = "MIP"  # default startup mode
    channels = read_channel_names(zpath)
    is_multichannel = len(channels) > 1

    def _add_single_image_layer(kind: str) -> Tuple[Optional[str], Tuple[int, int, int]]:
        """
        Add exactly ONE image layer ('main_image') of the requested type (MIP or 3D),
        set dims ndisplay appropriately, and return (z_key, scale_vec).
        If MIP requested but missing, warn and fall back to 3D.
        """
        # Remove any prior main image if present
        if "main_image" in viewer.layers:
            viewer.layers.remove("main_image")

        if kind == "MIP":
            if mip_zarr is None:
                show_warning("Requested MIP is not available; falling back to 3D.")
                return _add_single_image_layer("3D")
            arr = mip_zarr
            chosen_path = mip_path
            channel_axis = 1
            scale_vec_local = (1, 1, 1)
            z_key_local = None  # MIP doesn't require z in tracks if not present
            viewer.dims.ndisplay = 2

        elif kind == "3D":
            if image_zarr is None:
                show_warning("Requested 3D image is not available.")
                raise RuntimeError("No 3D image available.")
            arr = image_zarr
            chosen_path = zpath
            channel_axis = 1
            scale_vec_local = (3, 1, 1)
            z_key_local = "z"
            viewer.dims.ndisplay = 3
        else:
            raise ValueError(f"Unknown image kind: {kind}")

        # Optional contrast limits (endpoints-only per your flag)
        limits = None
        ch_names = read_channel_names(chosen_path)

        if USE_CONTRAST_SAMPLING:
            arr_limits = _arr_for_contrast(arr)
            limits = robust_contrast_limits_per_channel(
                arr_limits,
                multichannel=is_multichannel,
                p_low=P_LOW,
                p_high=P_HIGH,
                random_voxels_count=RANDOM_VOXELS_COUNT,
            )

        n_channels = int(arr.shape[channel_axis])
        if not ch_names or len(ch_names) != n_channels:
            ch_names = [f"ch{c}" for c in range(n_channels)]
        cmaps = [default_colormap_for_channel(ch_names[c], c) for c in range(n_channels)]

        img_kwargs = dict(
            channel_axis=channel_axis,
            scale=scale_vec_local,
            name="main_image",
            colormap=cmaps,
            visible=True,
        )
        if limits is not None:
            img_kwargs["contrast_limits"] = limits

        viewer.add_image(arr, **img_kwargs)
        return z_key_local, scale_vec_local

    # Initialize: ONE image layer (default MIP; warn/fallback handled inside)
    z_key, scale_vec = _add_single_image_layer(current_mode)

    # Add the "All tracks" layer, using single-source-of-truth z-key logic
    add_all_tracks_layer(
        viewer,
        track_dfs,
        current_mode=current_mode,
        mip_zarr=mip_zarr,
        primary_set=PRIMARY_TRACK_SET,
        thin_t_step=ALL_THIN_T_STEP,
        z_fallback="z",
        layer_name="All tracks",
        scale=scale_vec,
        tail_len=ALL_TAIL_LENGTH,
        tail_w=ALL_TAIL_WIDTH,
        color_by_default=default_color_by if default_color_by in colorables else "track_id",
        colormap_default="viridis",
    )

    # ---------------------- Mode Switch Widget (MIP <-> 3D) ----------------------
    @magicgui(
        mode={"choices": [m for m in ("MIP", "3D")
                          if (m == "MIP" and mip_zarr is not None) or (m == "3D" and image_zarr is not None)],
              "label": "Mode"},
        call_button="Apply",
    )
    def mode_dim_widget(mode=current_mode):
        nonlocal z_key, scale_vec, current_mode
        current_mode = mode

        # Remove dynamic track layers
        for lyr in list(viewer.layers):
            if lyr.name in ("selected track", "nearby tracks", "ROI tracks"):
                viewer.layers.remove(lyr)

        # Rebuild the ONE image layer for the selected mode
        z_key, scale_vec = _add_single_image_layer(mode)

        # Rebuild the "All tracks" in matching dimensionality using the single place for z_key logic
        if "All tracks" in viewer.layers:
            viewer.layers.remove("All tracks")
        add_all_tracks_layer(
            viewer,
            track_dfs,
            current_mode=current_mode,
            mip_zarr=mip_zarr,
            primary_set=PRIMARY_TRACK_SET,
            thin_t_step=ALL_THIN_T_STEP,
            z_fallback="z",
            layer_name="All tracks",
            scale=scale_vec,
            tail_len=ALL_TAIL_LENGTH,
            tail_w=ALL_TAIL_WIDTH,
            color_by_default=default_color_by if default_color_by in colorables else "track_id",
            colormap_default="viridis",
        )

    viewer.window.add_dock_widget(mode_dim_widget, area="right")

    # ------------------- Highlight Track & Nearby -------------------
    @magicgui(
        call_button="Highlight Track",
        layer_name={"choices": list(track_dfs.keys()), "label": "Track set"},
        track_id={"label": "Track ID"},
        dist_lim_um={"label": "Nearby distance (µm)", "min": 1, "max": 200, "step": 1, "widget_type": "SpinBox"},
    )
    def highlight_track_widget(layer_name, track_id: str, dist_lim_um: int = 90, color_by: Optional[str] = None, colormap: str = "viridis"):
        for lyr in list(viewer.layers):
            if lyr.name in ("selected track", "nearby tracks"):
                viewer.layers.remove(lyr)

        df = track_dfs.get(layer_name, None)
        if df is None or df.empty:
            show_warning(f"No tracks available for '{layer_name}'.")
            return

        # single-source-of-truth
        z_current = choose_z_key(df, current_mode, mip_zarr)

        df_sel = get_single_track_df(df, track_id, z_var=z_current)
        if df_sel is None or df_sel.empty:
            show_warning(f"Track ID {track_id} not found in {layer_name}.")
            return

        first_row = df_sel.iloc[0]
        t0 = int(first_row["t"])
        viewer.dims.set_current_step(0, t0)
        if z_current:
            try:
                viewer.dims.set_current_step(1, int(first_row[z_current]))
            except Exception:
                pass
        show_info(f"Moved viewer to t={t0}")

        add_tracks_with_coloring(
            viewer,
            df_sel,
            z_key=z_current,
            name="selected track",
            scale=scale_vec,
            tail_width=8,
            tail_length=40,
            opacity=1.0,
            color_by=(color_by or color_widget.color_by),
            colormap=(colormap or color_widget.colormap),
        )

        nearby = get_nearby_tracks(df, track_id, z_current, dist_um=dist_lim_um)
        if not nearby.empty:
            add_tracks_with_coloring(
                viewer,
                nearby,
                z_key=z_current,
                name="nearby tracks",
                scale=scale_vec,
                tail_width=2,
                tail_length=40,
                opacity=0.5,
                color_by=(color_by or color_widget.color_by) if (color_by or color_widget.color_by) in nearby.columns else None,
                colormap=(colormap or color_widget.colormap),
            )

    viewer.window.add_dock_widget(highlight_track_widget, area="right")

    # --------------------- ROI-based selection ---------------------
    def _ensure_roi_shapes_layer(viewer: napari.Viewer) -> "napari.layers.Shapes":
        if "ROI" in viewer.layers and isinstance(viewer.layers["ROI"], napari.layers.Shapes):
            return viewer.layers["ROI"]
        lyr = viewer.add_shapes(
            name="ROI",
            data=[],
            shape_type="polygon",
            edge_color="yellow",
            face_color='transparent',
            edge_width=1,
            opacity=0.4,
        )
        try:
            lyr.n_dimensional = False
            lyr.blending = "translucent_no_depth"
        except Exception:
            pass
        return lyr

    roi_layer = _ensure_roi_shapes_layer(viewer)

    def _collect_roi_polygons() -> List[np.ndarray]:
        polys: List[np.ndarray] = []
        try:
            for data, stype in zip(roi_layer.data, roi_layer.shape_type):
                if stype in ("polygon", "rectangle"):
                    a = np.asarray(data)
                    if a.ndim == 2 and a.shape[1] >= 2:
                        polys.append(a[:, :2])  # (y, x)
        except Exception as e:
            show_warning(f"ROI read error: {e}")
        return polys

    def _tracks_in_rois(
        df: pd.DataFrame,
        polygons: List[np.ndarray],
        z_key_local: Optional[str],
        t_mode: str,
        t0: Optional[int],
        t1: Optional[int],
        z_mode: str,
        z_tol_slices: int,
    ) -> np.ndarray:
        if df is None or df.empty or not polygons:
            return np.array([], dtype=int)

        if t_mode == "current":
            t = int(viewer.dims.current_step[0])
            sub = df[df["t"] == t]
        elif t_mode == "range" and (t0 is not None) and (t1 is not None):
            lo, hi = (int(min(t0, t1)), int(max(t0, t1)))
            sub = df[(df["t"] >= lo) & (df["t"] <= hi)]
        else:
            sub = df

        if z_mode == "current" and (z_key_local is not None) and (z_key_local in sub.columns):
            try:
                z_now = int(viewer.dims.current_step[1])
                sub = sub[np.abs(sub[z_key_local] - z_now) <= int(max(0, z_tol_slices))]
            except Exception:
                pass

        if sub.empty:
            return np.array([], dtype=int)

        xy = sub[["x", "y"]].to_numpy()
        inside_any = np.zeros(len(sub), dtype=bool)
        if MplPath is None:
            show_warning("matplotlib is required for ROI selection.")
            return np.array([], dtype=int)

        for poly in polygons:
            path = MplPath(np.c_[poly[:, 1], poly[:, 0]])  # (x, y)
            inside_any |= path.contains_points(xy)

        return sub.loc[inside_any, "track_id"].astype(int).unique()

    @magicgui(
        call_button="Select tracks in ROI",
        layer_name={"choices": lambda w=None: list(track_dfs.keys()), "label": "Track set"},
        time_scope={"choices": ["current", "range", "any"], "label": "Time scope"},
        t0={"label": "t0 (if range)", "min": 0, "step": 1, "widget_type": "SpinBox"},
        t1={"label": "t1 (if range)", "min": 0, "step": 1, "widget_type": "SpinBox"},
        z_scope={"choices": ["current", "any"], "label": "Z scope"},
        z_tol={"label": "Z tol (slices)", "min": 0, "max": 20, "step": 1, "widget_type": "SpinBox"},
        make_layer={"label": "Create 'ROI tracks' layer", "widget_type": "CheckBox"},
    )
    def roi_select_widget(
        layer_name: str,
        time_scope: str = "current",
        t0: int = 0,
        t1: int = 0,
        z_scope: str = "current",
        z_tol: int = 2,
        make_layer: bool = True,
    ):
        polygons = _collect_roi_polygons()
        if not polygons:
            show_warning("Draw at least one polygon/rectangle in the 'ROI' layer.")
            return

        df = track_dfs.get(layer_name, None)
        if df is None or df.empty:
            show_warning(f"No tracks available for '{layer_name}'.")
            return

        # single-source-of-truth
        z_current = choose_z_key(df, current_mode)

        tids = _tracks_in_rois(
            df=df,
            polygons=polygons,
            z_key_local=z_current,
            t_mode=time_scope,
            t0=t0 if time_scope == "range" else None,
            t1=t1 if time_scope == "range" else None,
            z_mode=z_scope,
            z_tol_slices=int(z_tol),
        )
        n = len(tids)
        if n == 0:
            show_info("No tracks found in ROI under current constraints.")
            return

        show_info(f"Selected {n} tracks in ROI.")
        if make_layer:
            roi_df = df[df["track_id"].isin(tids)].copy()
            if roi_df.empty:
                show_info("No rows to display for ROI selection.")
                return
            if "ROI tracks" in viewer.layers:
                viewer.layers.remove("ROI tracks")
            add_tracks_with_coloring(
                viewer,
                roi_df,
                z_key=z_current,
                name="ROI tracks",
                scale=scale_vec,
                tail_width=4,
                tail_length=40,
                opacity=0.9,
                color_by=color_widget.color_by,
                colormap=color_widget.colormap,
            )

    viewer.window.add_dock_widget(roi_select_widget, name="ROI Select", area="right")

    napari.run()


# ------------------------------ Entrypoint ------------------------------

if __name__ == "__main__":
    root = r"E:\Nick\killi_immuno_paper"
    project_name = "20241126_LCP1-NLSMSC" #"20250311_LCP1-NLSMSC" #"20250419_BC1-NLSMSC"  #"20250311_LCP1-NLSMSC" # #
    config = "tracking_lcp_nuclei"
    mip_flag = True
    main(root=root, project_name=project_name, tracking_config=config, mip_flag=mip_flag)
