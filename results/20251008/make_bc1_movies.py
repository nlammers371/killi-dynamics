from src.build_yx1.make_field_plots_v2 import healpix_to_mp4_v2
from pathlib import Path


if __name__ == '__main__':

    root = Path("/media/nick/cluster/projects/data/killi_tracker/")
    project_list = ["20250621", "20250716", "20250731"]
    well_list = [None, None, None]
    channel = [0, 0, 0]
    dt_vec = [47/60, 73/60, 68/60]
    dt_target = 0.75
    proj_mode = "mean" #"mean"
    overwrite = True
    nside = 128
    sigma_t = 0.75
    sigma_deg = 8
    deg_vec = [65, 60, 75]
    n_workers = 8

    for p, proj in enumerate(project_list[:-1]):
        healpix_to_mp4_v2(
            root=root,
            project_name=proj,
            wells=well_list[p],
            values_key=proj_mode,
            channel=channel[p],
            nside=128,
            overwrite=overwrite,
            dpi=600,
            dt_hours=dt_vec[p],
            target_dt_hours=dt_target,
            n_jobs=n_workers,
            temporal_sigma_hours=sigma_t,
            plot_kwargs=dict(
                bins=256,
                cmap="turbo",
                hemisphere="south",
                fov_deg=deg_vec[p],
                smooth_fwhm_deg=sigma_deg,
                rescale_to_fill=True,
            ),
        )