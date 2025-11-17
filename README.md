# Killi Dynamics

Tools for processing lightsheet and spinning-disk microscopy of *Danio rerio* embryos. The repository collects data-ingestion
utilities, segmentation and tracking pipelines, and higher-level cell-dynamics analysis in a single, reproducible codebase.【F:docs/pipeline_overview.md†L1-L78】

## Getting started

### Prerequisites
- Python 3.10+ with. GPU is recommended for segmentation and viewing functionality. 
- Bio-Formats compatible readers (`bioio`, `nd2`, `zarr`, `dask`) and scientific Python staples (`numpy`, `pandas`, `scipy`).
- Raw ND2 or CZI acquisitions stored under a project root that also contains `built_data/` and `metadata/` sub-directories, as
  expected by the exporters and downstream pipelines.【F:src/data_io/czi_export.py†L134-L200】【F:src/data_io/nd2_export.py†L64-L118】

> **Tip:** Keep raw data and generated Zarr stores on a high-throughput filesystem—segmentation and tracking read/write multi-gigabyte
> arrays per project.

### Installation
1. Create and activate a virtual environment (Conda, `venv`, or Poetry).
2. Install the repository in editable mode together with the GPU stack you need:
   ```bash
   pip install -e .
   ```
3. Install optional tools (Napari, Ultrack) if you plan to inspect masks or run tracking locally.

### Repository layout
- `docs/` – Lightweight design notes, including the updated [pipeline overview](docs/pipeline_overview.md) and [source
  inventory](docs/src_inventory.md).【F:docs/pipeline_overview.md†L1-L152】【F:docs/src_inventory.md†L1-L56】
- `src/data_io/` – CZI and ND2 exporters that produce Zarr stores and attach experiment metadata.【F:src/data_io/czi_export.py†L1-L200】【F:src/data_io/nd2_export.py†L1-L121】
- `src/registration/` – Hemisphere registration and on-disk fusion routines for dual-sided lightsheet experiments.【F:src/registration/register_hemispheres.py†L10-L124】【F:src/registration/fuse_on_disk.py†L10-L108】
- `src/segmentation/` – Cellpose/Li-threshold segmentation entry points and helpers for allocating mask/probability stores.【F:src/segmentation/cellpose.py†L28-L200】
- `src/qc/` – Volume, morphology, and surface-based filters that create cleaned mask datasets for tracking.【F:src/qc/mask_qc.py†L56-L180】
- `src/tracking/` – Ultrack orchestration plus utilities for relabelling masks and saving detection/segment outputs.【F:src/tracking/core_tracking.py†L15-L150】
- `src/geometry/`, `src/calculate_cell_fields/`, `src/cell_dynamics/` – Geometry fitting, HEALPix field computation, and
  higher-order cell-dynamics analysis built on top of tracked trajectories.【F:src/geometry/sphere.py†L17-L195】【F:src/calculate_cell_fields/calculate_fields.py†L18-L188】【F:src/cell_dynamics/pipeline.py†L47-L164】
- `src/pipelines/` – Command-line wrappers for chaining the major stages; start with the `lightsheet_cli` example.【F:src/pipelines/lightsheet_cli.py†L12-L82】
- `src/visualization/` – Napari and plotting scripts for manual QA of segmentation and threshold choices.【F:src/visualization/zarr_to_napari.py†L1-L33】
- `src/_Archive/` – Legacy scripts kept for reference; avoid importing them from new work.【F:docs/src_inventory.md†L6-L20】

### Typical processing workflow
1. **Export raw data:** Use the ND2 or CZI exporter to convert microscope files into Zarr. For example:
   ```python
   from src.export.nd2_export import export_nd2_to_zarr
   export_nd2_to_zarr(root="/data/killi", experiment_date="20240501", overwrite_flag=False, num_workers=4)
   ```
2. **Register hemispheres (dual-sided lightsheet):** Estimate per-frame shifts and write fusion metadata.
   ```python
   from src.registration.register_hemispheres import get_hemisphere_shifts
   get_hemisphere_shifts("/data/killi/built_data/zarr_image_files/20240501.zarr")
   ```
   Afterwards you can blend the volumes with `fuse_on_disk.fuse_hemisperes_on_disk` if needed.【F:src/registration/register_hemispheres.py†L39-L124】【F:src/registration/fuse_on_disk.py†L40-L108】
3. **Run segmentation:** Invoke the Cellpose pipeline to write mask/probability Zarr stores.
   ```python
   from src.segmentation.cellpose import cellpose_segmentation
   cellpose_segmentation(root="/data/killi", experiment_date="20240501", model_type="nuclei")
   ```
4. **Quality control:** Clean the masks with the QC wrapper.
   ```python
   from src.qc import mask_qc_wrapper
   mask_qc_wrapper("/data/killi", "20240501")
   ```
5. **Tracking:** Launch Ultrack through the CLI or directly via `perform_tracking` to produce track CSVs and segment stores.【F:src/tracking/core_tracking.py†L15-L150】
6. **Geometry and fields:** Fit embryo spheres and compute per-cell fields for downstream modelling.【F:src/geometry/sphere.py†L124-L195】【F:src/calculate_cell_fields/calculate_fields.py†L156-L188】
7. **Cell-dynamics analysis:** Run the `cell_dynamics.pipeline.run` entry point to assemble vector fields, MSD metrics, and QC summaries.【F:src/cell_dynamics/pipeline.py†L47-L164】

### Running the lightsheet CLI
For a scripted end-to-end run, call the lightsheet pipeline CLI (dry-run by default):
```bash
python -m src.pipelines.lightsheet_cli /data/killi 20240501 nuclei tracking_v1 --start 0 --stop 300 --execute
```
The CLI chains mask QC, Ultrack, and sphere fitting while printing progress so you can route it through a scheduler if needed.【F:src/pipelines/lightsheet_cli.py†L12-L82】

### Visualization
Launch Napari sessions directly from the provided scripts to spot-check probabilities, masks, and fused volumes:
```python
python -m src.visualization.zarr_to_napari
```
The viewer respects voxel scaling so spatial measurements remain meaningful.【F:src/visualization/zarr_to_napari.py†L1-L33】

## Next steps
- Review the [pipeline overview](docs/pipeline_overview.md) for deeper cross-references between modules.【F:docs/pipeline_overview.md†L1-L152】
- Use the notebooks in `notebooks/` for exploratory QC and plotting once deterministic steps have shipped into the CLI entry points.
- Log issues and roadmap items in `docs/src_reorg_roadmap.md` as the refactor progresses.
