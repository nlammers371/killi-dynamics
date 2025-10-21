# Pipeline overview

This guide summarizes how the active `src/` modules cooperate to move raw microscope acquisitions through segmentation, tracking,
and downstream analysis. Use it as a quick map when deciding where to plug in new code or when hunting for existing behaviour.

## Stage-by-stage directory map

| Stage | Key modules | Responsibilities |
| --- | --- | --- |
| Data ingestion | `src/data_io/czi_export.py`, `src/data_io/nd2_export.py`, `src/build_yx1/export_nd2_to_zarr.py` | Convert CZI/ND2 acquisitions into chunked Zarr stores, handle channel selection, resampling, and compatibility shims for older notebooks. |
| Preprocessing & projections | `src/image_utils/do_mip_projections.py` | Produce single- and dual-sided maximum-intensity projections used for quick inspection or QC overlays. |
| Segmentation | `src/segmentation/cellpose.py`, `src/segmentation/li_thresholding.py`, `src/segmentation/postprocess.py` | Run Cellpose across wells, manage Li-threshold alternatives, and consolidate masks/flows into Zarr outputs with metadata parity. |
| Quality control | `src/qc/mask_qc.py`, `src/qc/morphology.py`, `src/qc/shadows.py`, `src/qc/volumes.py`, `src/qc/surf.py` | Filter segmented nuclei by volume, shadowing, eccentricity, and surface distance while writing cleaned masks back to storage. |
| Tracking | `src/tracking/core_tracking.py`, `src/tracking/workflow.py`, `src/tracking/track_utils.py` | Prepare Ultrack inputs, launch tracking jobs, remap labels, and persist track CSVs/segment Zarr volumes. |
| Geometry & field analysis | `src/geometry/sphere.py`, `src/calculate_cell_fields/calculate_fields.py`, `src/build_yx1/surface_stats.py` | Fit embryo spheres, build HEALPix meshes, smooth spherical fields, and compute per-frame geometric descriptors. |
| Visualization & review | `src/visualization/inspect_masks.py`, `src/visualization/zarr_to_napari.py`, `src/visualization/check_thresholds.py` | Spin up Napari sessions and quick plots for manual QA of segmentation and threshold choices. |
| Modeling | `src/vae/` | Autoencoder configs, trainers, and evaluation scripts for latent-trajectory experiments. |
| Orchestration | `src/pipelines/lightsheet_cli.py` | CLI wrapper that chains QC, Ultrack, and sphere fitting for lightsheet datasets. |
| Active utilities | `src/utilities/functions.py` | Minimal helper surface kept for code still in rotation. |
| Archives | `src/_Archive/` (including the new `utilities/` subfolder) | Legacy modules retained for reference; imports should be avoided in new work. |

## Highlights by module

### Data ingestion
- **`czi_export.py`** inspects Bio-Formats metadata, determines multi-timepoint versus list exports, and initializes/resumes Zarr stores with the requested resampling scale and channels before streaming chunks that still need writing.【F:src/data_io/czi_export.py†L36-L158】
- **`nd2_export.py`** wraps Nikon ND2 exports by collecting experiment metadata, permuting axes into a Dask array, and writing each well (and optional MIP) into structured Zarr groups with metadata copied alongside.【F:src/data_io/nd2_export.py†L11-L118】
- **`build_yx1/export_nd2_to_zarr.py`** remains as a deprecated shim that re-exports the modern ND2 helpers so older notebooks keep working.【F:src/build_yx1/export_nd2_to_zarr.py†L1-L25】

### Preprocessing & projections
- **`image_utils/do_mip_projections.py`** loads fused or single-sided volumes, rechunks them for Dask processing, emits dual- or single-sided maximum intensity projections, and saves the result to Zarr while preserving attributes.【F:src/image_utils/do_mip_projections.py†L10-L112】

### Segmentation
- **`segmentation/cellpose.py`** exposes both single-field `segment_fov` runs and the full experiment `cellpose_segmentation` routine, handling curation metadata, anisotropy, Zarr allocation for masks/probabilities, and model bookkeeping.【F:src/segmentation/cellpose.py†L1-L200】
- Complementary modules such as **`li_thresholding.py`** and **`postprocess.py`** provide traditional thresholding and cleanup stages invoked by the same pipelines.

### Quality control
- **`qc/mask_qc.py`** orchestrates per-frame filtering by volume, shadowing, eccentricity, and surface distance, writing cleaned masks and exposing a `mask_qc_wrapper` entry point consumed by the pipeline CLI.【F:src/qc/mask_qc.py†L41-L167】【F:src/qc/__init__.py†L1-L20】

### Tracking
- **`tracking/core_tracking.py`** prepares mask stacks, derives Ultrack detection/boundary inputs, and saves tracked segments and CSV outputs for configurable frame ranges.【F:src/tracking/core_tracking.py†L15-L150】
- **`tracking/workflow.py`** adds helpers for copying frames, relabeling masks, and writing Dask outputs, plus higher-level orchestration for multiprocessing-friendly tracking workflows.【F:src/tracking/workflow.py†L1-L46】

### Geometry & field analysis
- **`geometry/sphere.py`** fits per-frame embryo spheres, generates meshes, and smooths center trajectories before exporting per-well CSV summaries.【F:src/geometry/sphere.py†L1-L195】
- **`calculate_cell_fields/calculate_fields.py`** supplies spherical smoothing, velocity derivations, and neighbour-degree metrics used to interpret tracked trajectories on the embryo surface.【F:src/calculate_cell_fields/calculate_fields.py†L1-L120】
- **`build_yx1/surface_stats.py`** retains HEALPix-to-mesh and map projection utilities leveraged by the field calculations.【F:src/build_yx1/surface_stats.py†L1-L66】

### Visualization & review
- **`visualization/zarr_to_napari.py`** demonstrates how to open segmentation probabilities in Napari with voxel scaling for frame-by-frame QA, while related scripts stage other inspection entry points.【F:src/visualization/zarr_to_napari.py†L1-L29】

### Modeling
- **`src/vae/`** houses latent-model configuration, training loops, and evaluation utilities that operate on the curated trajectory outputs (see the package-level README and module structure under `models/`, `trainers/`, and `pipelines/`).【F:src/vae/trainers/base_trainer/base_trainer.py†L1-L40】

### Utilities and archive
- **Active utilities** now only expose `path_leaf` for file-name extraction, keeping the shared helper surface minimal for active code.【F:src/utilities/functions.py†L1-L12】
- **Legacy helpers**—including Zarr IO wrappers, plotting helpers, geometric utilities, and ND2 metadata shims—have been relocated to `src/_Archive/utilities/` for safe keeping without polluting the active namespace.【F:src/_Archive/utilities/functions.py†L1-L30】【F:src/_Archive/utilities/io.py†L1-L170】
