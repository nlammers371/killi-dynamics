# `src` Directory Guide

This repository is migrating from script-centric folders toward domain-focused packages. The goal is to make it easier to find
related functionality (exports, segmentation, QC, tracking, geometry, and orchestration) while we gradually move individual
modules. During the transition:

- Existing script folders such as `build_lightsheet/`, `build_yx1/`, and `nucleus_dynamics/` continue to host the actual
  implementations.
- New top-level packages under `src/` provide the public import surface that downstream code should target.
- Each package exposes the same functions currently used in pipelines so that imports can change now without moving the code.

## Target layout

```
src/
├── data_io/          # microscope exports, metadata extraction, and format helpers
├── segmentation/     # thresholding, mask generation, and segmentation post-processing
├── qc/               # reusable mask and track quality-control predicates
├── tracking/         # Ultrack configuration, relabeling helpers, and track-merging utilities
├── geometry/         # sphere fitting, spherical harmonics, and coordinate transforms
└── pipelines/        # high-level orchestration entry points (lightsheet, ND2, etc.)
```

As modules are extracted, the new packages will absorb their implementations. For now they serve as clearly named waypoints so
that the larger refactor can proceed in small, reviewable increments.

## How users should run the pipelines

While the codebase stabilizes, favor a two-tier workflow:

- **Command-line entry points** (e.g., `src/pipelines/lightsheet_cli.py`) handle deterministic export/segment/track jobs using
  configuration files for reproducibility.
- **Curated Jupyter notebooks** remain the venue for exploratory QC or visualization tasks and can call the CLI when deterministic
  steps are needed.

See [`docs/pipeline_entrypoints.md`](../docs/pipeline_entrypoints.md) for the rationale and implementation roadmap.
