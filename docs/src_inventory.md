# Current `src` Inventory

Snapshot of the active, post-reorg layout (excluding `var/`). The first two levels are expanded so reviewers can spot the
major packages without wading through implementation details.

````
src/
    ├── _Archive/
        ├── build_lightsheet/
        ├── nucleus_dynamics/
        ├── track_processing/
        ├── utilities/
        └── vae/
    ├── build_yx1/
        ├── export_nd2_to_zarr.py
        ├── fit_embryo_surface.py
        ├── make_field_plots.py
        ├── make_field_plots_v2.py
        ├── project_density_fields.py
        ├── project_scalar_fields.py
        ├── surface_stats.py
    ├── calculate_cell_fields/
        ├── build_nucleus_data.py
        ├── calculate_fields.py
        ├── cluster_tracking.py
        ├── density_functions.py
        ├── plotting.py
    ├── cell_dynamics/
        ├── adaptivity.py
        ├── cd_utils.py
        ├── config.py
        ├── flux.py
        ├── grids.py
        ├── io_functions.py
        ├── materials.py
        ├── metrics.py
        ├── msd.py
        ├── pipeline.py
        ├── qc.py
        ├── vector_field.py
    ├── data_io/
        ├── __init__.py
        ├── czi_export.py
        ├── nd2_export.py
        └── _Archive/
    ├── geometry/
        ├── __init__.py
        ├── sphere.py
        └── spherical_harmonics.py
    ├── image_utils/
        └── do_mip_projections.py
    ├── pipelines/
        ├── __init__.py
        └── lightsheet_cli.py
    ├── qc/
        ├── mask_qc.py
        ├── morphology.py
        ├── shadows.py
        ├── surf.py
        └── volumes.py
    ├── registration/
        ├── fuse_on_disk.py
        ├── register_hemispheres.py
        ├── virtual_fusion.py
        └── _Archive/
    ├── segmentation/
        ├── cellpose.py
        ├── li_thresholding.py
        ├── mask_builders.py
        ├── postprocess.py
        └── segmentation_wrappers.py
    ├── tracking/
        ├── core_tracking.py
        ├── track_utils.py
        └── workflow.py
    ├── utilities/
        └── functions.py
    ├── visualization/
        ├── check_thresholds.py
        ├── inspect_masks.py
        ├── zarr_to_napari.py
        └── _Archive/
    ├── __init__.py
    └── README.md
````
