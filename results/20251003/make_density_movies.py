from src.build_yx1.make_field_plots import healpix_to_mp4
from pathlib import Path


if __name__ == '__main__':

    root = Path("/media/nick/cluster/projects/data/killi_tracker/")
    project_list = ["20250621", "20250716", "20250731"]
    well_list = [None]
    channel = [1, 1, 0]
    proj_mode = "density" #"mean"
    overwrite = True
    nside=128
    deg_vec = [65, 60, 75]

    for p, proj in enumerate(project_list):
        healpix_to_mp4(
            root=root,
            project_name=proj,
            values_key=proj_mode,
            channel=channel[p],
            nside=128,
            overwrite=overwrite,
            dpi=600,
            mode="png",
            plot_kwargs=dict(
                bins=256,
                cmap="magma",
                hemisphere="south",
                fov_deg=deg_vec[p],
                smooth_fwhm_deg=16,
                rescale_to_fill=True,
            ),
        )