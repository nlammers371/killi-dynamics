from src.build_yx1.make_field_plots import healpix_to_mp4



if __name__ == '__main__':
    healpix_to_mp4(
        "well0001_field.zarr",
        "movies/well0001.mp4",
        mode="mp4",
        fps=15,
        cleanup=True,
        plot_kwargs=dict(
            cmap="turbo",
            channel=0,
            values_key="max",
            hemisphere="south",
            fov_deg=80,
            smooth_fwhm_deg=8,
            rescale_to_fill=True,
            dpi=600,
        ),
    )