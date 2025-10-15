# Pipeline interaction strategy

To keep the reorganization manageable while giving users a clear on-ramp, prefer a two-tier interface:

1. **Command-line workflows for routine builds.**
   - Wrap each major pipeline (lightsheet, ND2/YX1, tracking-only, QC reports) in a thin `typer` or `click` CLI that accepts an input config (YAML/TOML) describing datasets, storage paths, and optional stages.
   - Use the upcoming `src/pipelines/` package to collect these orchestrators so automation and scheduled runs can invoke them reproducibly (e.g., `python -m src.pipelines.lightsheet build --config config.yaml`).
   - Emit structured logs/progress bars and hand back paths to generated artifacts; this makes it easy to hand off long runs to a cluster or Slurm job.

2. **Curated Jupyter notebooks for exploratory or QC-heavy tasks.**
   - Keep notebooks focused on interactive analysis, visualization, or one-off debugging where human judgment is required (e.g., adjusting segmentation thresholds, inspecting tracking merges).
   - Whenever a notebook step becomes deterministic, graduate it into the CLI pipeline and have the notebook call the CLI (via `subprocess` or `%run`) to avoid code drift.

## Why not a single "master" script?

A monolithic script quickly becomes brittle as optional stages multiply. Modular CLIs let you run only the pieces you need, chain them in schedulers, and keep responsibilities narrow. They also dovetail with configuration-driven pipelines, which aids reproducibility.

## Implementation roadmap

1. **Define shared config schemas** inside `src/pipelines/config.py` (to be added) so both CLI tools and notebooks load the same structured settings.
2. **Refactor existing scripts into callable functions** (e.g., `build_lightsheet_dataset(config)`) and expose them via CLI entry points.
3. **Add notebook templates** that demonstrate calling the CLI and visualizing outputs, emphasizing that notebooks are for inspection, not orchestration.

This split gives lab members a dependable automation path while retaining the flexibility of interactive exploration when needed.
