from src.build_yx1.make_field_plots_v2 import healpix_to_mp4_v2
from pathlib import Path


if __name__ == '__main__':

    root = Path("/media/nick/cluster/projects/data/killi_tracker/")
    project_list = ["20250621", "20250716", "20250731"]
    well_list = [[0]]
    channel = [1, 1, 0]
    dt_vec = [45/60, 73/60, 68/60]
    dt_target = 0.5
    proj_mode = "density" #"mean"
    overwrite = True
    nside = 128
    deg_vec = [65, 60, 75]

    for p, proj in enumerate(project_list):
        healpix_to_mp4_v2(
            root=root,
            project_name=proj,
            values_key=proj_mode,
            channel=channel[p],
            wells=None,
            nside=128,
            overwrite=overwrite,
            dpi=600,
            mode="png",
            dt_hours=dt_vec[p],
            target_dt_hours=dt_target,
            temporal_sigma_hours=1,
            plot_kwargs=dict(
                bins=256,
                cmap="turbo",
                hemisphere="south",
                fov_deg=deg_vec[p],
                smooth_fwhm_deg=10,
                rescale_to_fill=True,
            ),
        )