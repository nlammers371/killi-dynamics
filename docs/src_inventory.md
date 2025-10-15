# Current `src` Inventory

Snapshot of the pre-refactor layout (excluding `var/`). Only the first two levels are expanded to keep the view reviewable.

````
src/
    ├── _Archive/
        ├── core_tracking/
        ├── __init__.py
    ├── build_lightsheet/
        ├── __init__.py
        ├── build_utils.py
        ├── fit_embryo_surface.py
        ├── fuse_masks.py
        ├── nucleus_classification.py
        ├── process_masks.py
        ├── run00_get_frame_shifts.py
        ├── run01_get_hemisphere_shifts.py
        ├── run02_segment_nuclei.py
        ├── stitch_image_stacks.py
        ├── track_processing.py
    ├── build_yx1/
        ├── __init__.py
        ├── export_nd2_to_zarr.py
        ├── fit_embryo_surface.py
        ├── make_field_plots.py
        ├── make_field_plots_v2.py
        ├── project_density_fields.py
        ├── project_scalar_fields.py
        ├── surface_stats.py
    ├── data_io/
        ├── __init__.py
    ├── geometry/
        ├── __init__.py
    ├── image_utils/
        ├── do_mip_projections.py
    ├── nucleus_dynamics/
        ├── build/
        ├── export_to_zarr/
        ├── tracking/
        ├── utilities/
    ├── pipelines/
        ├── __init__.py
    ├── qc/
        ├── __init__.py
    ├── segmentation/
        ├── __init__.py
    ├── symmetry_breaking/
        ├── cluster_tracking.py
        ├── density_functions.py
    ├── track_processing/
        ├── filter_tracks.py
    ├── tracking/
        ├── __init__.py
    ├── utilities/
        ├── __init__.py
        ├── convert_to_zarr.py
        ├── extract_frame_metadata.py
        ├── functions.py
        ├── image_utils.py
        ├── io.py
        ├── plot_functions.py
        ├── plot_utils.py
        ├── register_image_stacks.py
        ├── shape_utils.py
    ├── vae/
        ├── auxiliary_scripts/
        ├── models/
        ├── pipelines/
        ├── trainers/
        ├── __init__.py
        ├── config.py
        ├── customexception.py
    ├── __init__.py
    ├── README.md
````
