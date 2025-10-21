# Pipeline Overview

This document summarizes the active pieces of the imaging pipeline that now live under `src/`.  It is ordered in the way a lightsheet experiment typically progresses, with links to the subpackages that provide each stage of the workflow.

## 1. Data ingestion and registration
- **`data_io/`** – Home for importers that convert microscope exports to OME-Zarr, including `czi_export.py` for lightsheet `.czi` data and `nd2_export.py` for spinning disk `.nd2` data.  Metadata helpers such as `nd2_metadata.py` live alongside the exporters.
- **`build_lightsheet/run00_get_frame_shifts.py` & `run01_get_hemisphere_shifts.py`** – Utilities for estimating coarse rigid shifts between time points or embryo halves prior to segmentation.
- **`nucleus_dynamics/utilities/`** – Contains shared routines (e.g. `image_utils.calculate_LoG`, `register_image_stacks.register_timelapse`) that the legacy Cellpose jobs still depend on while the codebase transitions to the new packages.

## 2. Segmentation
- **`segmentation/`** – Consolidates the Li-threshold and Cellpose-based segmentation stack into `thresholding.py`, `mask_builders.py`, `postprocess.py`, and `cellpose.py`.  The functions here operate on numpy arrays / zarr chunks and avoid side effects so that pipelines can orchestrate them cleanly.
- **`build_lightsheet/fuse_masks.py`** – Composes the segmentation primitives for multi-view lightsheet experiments, handles mask fusion across views, and prepares intermediate zarr datasets for QC.

## 3. Quality control
- **`qc/`** – Implements reusable filters for mask cleanup: `volumes.py` for size/intensity thresholds, `shadows.py` for lightsheet shadow detection, `morphology.py` for eccentricity / topology metrics, and `mask_qc.py` which wires them together.  Geometry-aware checks (e.g., distance-to-surface) now sit next to the filters instead of inside monolithic scripts.
- **`build_lightsheet/process_masks.py` (recently refactored)** – Calls the new QC primitives during dataset construction, leaving only orchestration and configuration logic in the pipeline layer.

## 4. Tracking
- **`tracking/workflow.py`** – Wraps Ultrack configuration, segmentation-to-track conversions, and result stitching.  It exposes functions such as `run_tracking`, `combine_tracking_results`, and `export_tracks` that downstream code can call without touching Ultrack internals.
- **`build_lightsheet/track_processing.py`** – Higher-level routines for merging tracking output back into the experiment layout (e.g., moving node tables into project folders, synchronizing metadata with geometry fits).

## 5. Geometry & embryo surface models
- **`geometry/`** – Provides `sphere.py` for fast least-squares sphere fitting and shell generation, plus `spherical_harmonics.py` for refined surface reconstructions used during QC and downstream analyses.
- **`build_lightsheet/fit_embryo_surface.py`** – Uses the geometry package to estimate embryo shells from QC-approved masks and persists parameters for later use in digital embryo reconstructions.

## 6. Orchestration and entry points
- **`pipelines/lightsheet_cli.py`** – A sample command-line interface demonstrating how to tie the modular pieces together: export raw data, segment, run QC, track, and fit geometry with shared configuration.
- **`build_lightsheet/build_utils.py`** – Houses shared helpers for the lightsheet pipeline (chunked zarr writers, metadata lookups, etc.) that are gradually being migrated into the focused packages above.

## 7. Supporting modules
- **`image_utils/do_mip_projections.py`** – Convenience routines for generating quick-look projections of time-lapse datasets.
- **`track_processing/`** – Additional track filtering and analysis scripts that consume the standardized output of `tracking.workflow`.
- **`utilities/functions.py`** – Retains only the widely used `path_leaf` helper; all other legacy utilities now reside under `src/_Archive/utilities/`.

## 8. Archived helpers
- **`src/_Archive/`** – Captures deprecated segmentation, QC, and geometry scripts along with the archived utility modules (`utilities/`) that were removed from active code.  Historical notebooks and analyses can continue importing from `src._Archive.utilities.*` without polluting the active namespace.

Together these packages describe a linear pipeline:
1. Export microscope volumes to Zarr (`data_io`).
2. Estimate rigid alignment / frame shifts (`build_lightsheet` utilities).
3. Generate nuclear masks with the segmentation primitives (`segmentation`).
4. Apply morphology, intensity, and geometry-aware QC filters (`qc`).
5. Track nuclei over time (`tracking`).
6. Fit embryo surfaces and spherical-harmonic shells for downstream analyses (`geometry`).
7. Drive the whole process via scripts in `build_lightsheet/` or CLI entry points under `pipelines/`.

This overview should help contributors locate functionality quickly and understand which modules to touch when editing a particular stage of the workflow.
