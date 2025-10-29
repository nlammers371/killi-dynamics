# Pipeline overview

This guide summarizes how the active `src/` modules cooperate to move raw microscope acquisitions through registration,
segmentation, tracking, and downstream analysis. Use it as a quick map when deciding where to plug in new code or when
hunting for existing behaviour.

## Stage-by-stage directory map

| Stage | Key modules | Responsibilities |
| --- | --- | --- |
| Data ingestion | `src/data_io/czi_export.py`, `src/data_io/nd2_export.py`, `src/build_yx1/export_nd2_to_zarr.py` | Convert CZI/ND2 acquisitions into chunked Zarr stores, optionally resample, and provide compatibility shims for historical notebooks.【F:src/data_io/czi_export.py†L1-L200】【F:src/data_io/nd2_export.py†L1-L121】【F:src/build_yx1/export_nd2_to_zarr.py†L1-L25】 |
| Registration & fusion | `src/registration/register_hemispheres.py`, `src/registration/fuse_on_disk.py` | Estimate per-frame rigid shifts between hemispheres and write fused volumes back to storage when dual-sided data is available.【F:src/registration/register_hemispheres.py†L10-L124】【F:src/registration/fuse_on_disk.py†L10-L108】 |
| Preprocessing & projections | `src/image_utils/do_mip_projections.py` | Generate single- or dual-sided maximum intensity projections for quick QC and visualization.【F:src/image_utils/do_mip_projections.py†L10-L112】 |
| Segmentation | `src/segmentation/cellpose.py`, `src/segmentation/li_thresholding.py`, `src/segmentation/postprocess.py`, `src/segmentation/segmentation_wrappers.py` | Run Cellpose or Li-threshold pipelines, manage metadata-aware mask storage, and expose helpers for downstream QC.【F:src/segmentation/cellpose.py†L1-L200】 |
| Quality control | `src/qc/mask_qc.py`, `src/qc/morphology.py`, `src/qc/shadows.py`, `src/qc/volumes.py`, `src/qc/surf.py` | Filter nuclei by volume, shadowing, morphology, and surface distance before writing cleaned masks into the shared Zarr stores.【F:src/qc/mask_qc.py†L1-L180】 |
| Tracking | `src/tracking/core_tracking.py`, `src/tracking/workflow.py`, `src/tracking/track_utils.py` | Prepare Ultrack inputs from cleaned masks, launch tracking jobs, and persist detections/segments alongside CSV summaries.【F:src/tracking/core_tracking.py†L1-L150】 |
| Geometry & field analysis | `src/geometry/sphere.py`, `src/calculate_cell_fields/calculate_fields.py`, `src/build_yx1/surface_stats.py` | Fit embryo spheres, construct spherical meshes/fields, and compute per-frame geometric descriptors to support downstream modelling.【F:src/geometry/sphere.py†L1-L195】【F:src/calculate_cell_fields/calculate_fields.py†L1-L188】【F:src/build_yx1/surface_stats.py†L1-L66】 |
| Cell-dynamics pipeline | `src/cell_dynamics/pipeline.py`, `src/cell_dynamics/vector_field.py`, `src/cell_dynamics/metrics.py`, `src/cell_dynamics/flux.py` | Smooth tracks, assemble HEALPix bins, compute vector and scalar metrics, and package results for notebook exploration or reporting.【F:src/cell_dynamics/pipeline.py†L47-L164】 |
| Visualization & review | `src/visualization/inspect_masks.py`, `src/visualization/zarr_to_napari.py`, `src/visualization/check_thresholds.py` | Launch Napari sessions or quick plots to inspect segmentation, thresholds, and probabilities without re-running pipelines.【F:src/visualization/zarr_to_napari.py†L1-L33】 |
| Orchestration | `src/pipelines/lightsheet_cli.py` | CLI wrapper that chains QC, tracking, and sphere fitting for lightsheet datasets (dry-run friendly).【F:src/pipelines/lightsheet_cli.py†L1-L82】 |
| Utilities & archive | `src/utilities/functions.py`, `src/_Archive/` | Keep the shared helper surface minimal while parking legacy scripts in `_Archive/` for reference only.【F:src/utilities/functions.py†L1-L13】 |

## Highlights by module

### Data ingestion
- **`czi_export.py`** inspects Bio-Formats metadata, detects single- versus multi-scene acquisitions, and initializes/resumes
  Zarr arrays with channel selection and anisotropy checks before streaming unfinished chunks.【F:src/data_io/czi_export.py†L1-L200】
- **`nd2_export.py`** wraps Nikon ND2 exports by permuting axes into Dask arrays and writing each well (plus optional MIPs)
  into structured Zarr groups with metadata copied alongside.【F:src/data_io/nd2_export.py†L1-L121】
- **`build_yx1/export_nd2_to_zarr.py`** remains as a thin compatibility shim that re-exports the modern ND2 helpers so older
  notebooks keep working.【F:src/build_yx1/export_nd2_to_zarr.py†L1-L25】

### Registration & fusion
- **`register_hemispheres.py`** aligns dual-sided acquisitions by sampling frames, running phase cross correlation, and
  interpolating per-frame shifts that are written to the Zarr metadata for later fusion.【F:src/registration/register_hemispheres.py†L10-L124】
- **`fuse_on_disk.py`** consumes those shifts to blend the two hemispheres frame by frame, updating fused arrays and metadata
  in-place within the same store.【F:src/registration/fuse_on_disk.py†L10-L108】

### Preprocessing & projections
- **`image_utils/do_mip_projections.py`** rechunks Zarr-backed volumes for efficient Dask processing, produces single- or dual
  sided maximum intensity projections, and writes the result back with preserved attributes.【F:src/image_utils/do_mip_projections.py†L55-L111】

### Segmentation
- **`segmentation/cellpose.py`** exposes both single-volume `segment_fov` helpers and the experiment-level `cellpose_segmentation`
  routine, handling curation metadata, anisotropy, mask/probability Zarr allocation, and model bookkeeping.【F:src/segmentation/cellpose.py†L28-L200】
- Complementary modules such as **`li_thresholding.py`** and **`postprocess.py`** provide traditional thresholding and cleanup
  stages invoked by the same pipelines.

### Quality control
- **`qc/mask_qc.py`** orchestrates per-frame filtering by volume, shadowing, eccentricity, and surface distance, writing cleaned
  masks and exposing a `mask_qc_wrapper` entry point consumed by the pipeline CLI.【F:src/qc/mask_qc.py†L56-L180】

### Tracking
- **`tracking/core_tracking.py`** prepares mask stacks, derives Ultrack detection/boundary inputs, launches tracking, and persists
  track CSVs/segment Zarr volumes for configurable frame ranges.【F:src/tracking/core_tracking.py†L15-L150】
- **`tracking/workflow.py`** and **`tracking/track_utils.py`** retain supporting helpers for copying frames, relabelling masks, and
  converting label images into Ultrack detections.

### Geometry & field analysis
- **`geometry/sphere.py`** fits per-frame embryo spheres, generates meshes, and smooths center trajectories before exporting per-well
  CSV summaries used downstream.【F:src/geometry/sphere.py†L17-L195】
- **`calculate_cell_fields/calculate_fields.py`** supplies smoothing, spherical projections, neighbour-degree metrics, and Zarr writing
  utilities that operate on the tracked trajectories and HEALPix meshes.【F:src/calculate_cell_fields/calculate_fields.py†L18-L188】
- **`build_yx1/surface_stats.py`** retains HEALPix-to-mesh and map projection utilities leveraged by the field calculations.【F:src/build_yx1/surface_stats.py†L1-L66】

### Cell-dynamics pipeline
- **`cell_dynamics/pipeline.py`** is the high-level entry point that loads tracking outputs, smooths trajectories, bins them on HEALPix grids,
  computes vector/scalar/MSD metrics, and writes augmented tables/Zarr stores for reporting.【F:src/cell_dynamics/pipeline.py†L47-L164】
- Supporting modules (`vector_field.py`, `metrics.py`, `flux.py`, `grids.py`, etc.) hold the numerical routines invoked by the pipeline.

### Visualization & review
- **`visualization/zarr_to_napari.py`** demonstrates how to open segmentation probabilities in Napari with voxel scaling for frame-by-frame QA,
  while related scripts stage other inspection entry points.【F:src/visualization/zarr_to_napari.py†L1-L33】

### Orchestration and utilities
- **`pipelines/lightsheet_cli.py`** provides an argparse-based CLI that chains mask QC, Ultrack tracking, and sphere fitting, doubling as a dry-run
  to preview the actions a configuration will take.【F:src/pipelines/lightsheet_cli.py†L12-L82】
- **`utilities/functions.py`** intentionally exposes only `path_leaf`, keeping the shared helper surface minimal for active code, while legacy
  helpers live in `_Archive/` for reference only.【F:src/utilities/functions.py†L1-L13】
