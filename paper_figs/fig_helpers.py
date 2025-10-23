# color_helpers.py
import numpy as np
import matplotlib
from napari.utils.colormaps import Colormap, AVAILABLE_COLORMAPS
import napari

def add_matplotlib_cmap_to_napari(name: str):
    """Convert and register a Matplotlib colormap in Napari."""
    if name not in matplotlib.colormaps:
        raise ValueError(f"{name} not found in Matplotlib colormaps")
    rgba = matplotlib.colormaps[name](np.linspace(0, 1, 256))
    cmap = Colormap(rgba, name=name)
    AVAILABLE_COLORMAPS[name] = cmap
    return cmap

def register_standard_colormaps():
    """Register and return a consistent set of colormaps."""
    names = ["RdYlBu", "RdYlGn", "Spectral", "PRGn"]
    return {n: add_matplotlib_cmap_to_napari(n) for n in names}

def make_bop_variants():
    """Return a consistent set of bop_* variants with white/transparent starts."""
    base_blue = AVAILABLE_COLORMAPS["bop blue"]
    base_orange = AVAILABLE_COLORMAPS["bop orange"]

    def _make_variant(base, name, start_white=False, trim_start=None):
        rgba = np.array(base.colors)
        if trim_start:
            rgba = rgba[trim_start:, :]
        if start_white:
            rgba[0, :3] = [1, 1, 1]
        rgba[0, -1] = 0
        return Colormap(rgba, name=name)

    bop_blue_white = _make_variant(base_blue, "bop_blue_white", start_white=True, trim_start=64)
    bop_orange_white = _make_variant(base_orange, "bop_orange_white", start_white=True)

    for cmap in [bop_blue_white, bop_orange_white]:
        AVAILABLE_COLORMAPS[cmap.name] = cmap
    return {"bop_blue_white": bop_blue_white, "bop_orange_white": bop_orange_white}


