import numpy as np
import zarr, healpy as hp
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")   # or "TkAgg" if Qt isn't installed
import matplotlib.pyplot as plt
from src.build_yx1.make_field_plots import plot_healpix_equalarea_frame, get_intensity_bounds_from_last_frame


if __name__ == '__main__':

    root = "/media/nick/cluster/projects/data/killi_tracker/"
    project_name = "20250716"
    projection_path = Path(root) / "output_data" / "sphere_projections" / project_name
    well_list = sorted(projection_path.glob("*0128.zarr"))
    well_ind = 0
    well_zarr_path = well_list[well_ind]
    values_key="mean"
    t_index=32
    channel=0
    nside=None,

    sm_deg = 12
    mode = "density"

    vmin, vmax = get_intensity_bounds_from_last_frame(
        well_zarr_path,
        channel=0,
        values_key=mode,
        smooth_fwhm_deg=sm_deg)

    plot_healpix_equalarea_frame(zarr_path=well_zarr_path,
                                 t=t_index,
                                 channel=channel,
                                 values_key=mode,
                                 bins=256,
                                 vmin=vmin,
                                 vmax=vmax * 0.8,
                                 smooth_fwhm_deg=12, #None,
                                 fov_deg=75,
                                 rescale_to_fill=True,
                                 save_path="test.png",
                                 cmap="magma",)

