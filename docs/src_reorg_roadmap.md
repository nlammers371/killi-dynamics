# `src` Reorganization Roadmap

This roadmap sequences the restructuring work so that each step is reviewable yet still yields a useful unit of progress. Each milestone assumes prior steps are merged, keeping the diff surface focused.

## 0. Lay the groundwork
- Capture the current layout: freeze a short `docs/src_inventory.md` (auto-generated tree or table) so later refactors can be audited quickly.
- Add `src/README.md` describing the new target structure and naming conventions to align reviewers before code moves begin.

## 1. Carve out high-level domains
- **Create empty packages** (`data_io`, `segmentation`, `tracking`, `geometry`, `qc`, `pipelines`) with `__init__` stubs and module docstrings.
- Migrate only the *imports* in a few orchestration scripts to reference the new packages (via re-exports), but keep the implementation code in place. This makes the intent visible without risky mass moves.

## 2. Extract data IO modules
- Move CZI and ND2 exporters plus shared metadata helpers into `src/data_io/`.
- Update scripts/notebooks that consume these functions; keep backward-compatible import shims in the old locations for one release.
- Add smoke tests that load a representative config and run dry-run exports to lock in interfaces.

## 3. Segmentation split
- Relocate LI-threshold utilities, Cellpose wrappers, watershed/post-processing into submodules (`thresholding.py`, `mask_builders.py`, `postprocess.py`).
- Convert pipeline scripts to call the new segmentation API entry points.
- Document configuration options centrally (module-level docstrings or `CONFIG.md`).

## 4. Quality-control consolidation
- Pull QC predicates (size filters, shadow checks, ellipsoid fits) into `src/qc/` as composable functions.
- Refactor lightsheet QC notebooks/scripts to use these helpers; retire duplicated logic.
- Add regression-style tests on small sample masks to confirm filter behavior.

## 5. Tracking and geometry refactor
- Group Ultrack orchestration, relabeling, and track-merging under `src/tracking/`.
- Collect sphere/spherical-harmonics utilities into `src/geometry/` with clear IO interfaces.
- Where pipelines depend on both, introduce cohesive pipeline modules that orchestrate segmentation → QC → tracking → geometry in a readable order.

## 6. Pipeline wrappers and CLIs
- After core modules stabilize, introduce thin CLI/napari drivers under `src/pipelines/` or `src/interfaces/`.
- Ensure each pipeline has a canonical configuration file and a reproducible entry point.

## 7. Cleanup and deprecation removals
- Remove old compatibility shims once downstream notebooks/scripts are updated.
- Finalize documentation: update README, architecture diagrams, and usage guides.

### Tips for keeping reviews manageable
- Cap each PR to one domain move (e.g., "Extract CZI exporter into `data_io`").
- Provide before/after import matrices or dependency graphs in PR descriptions.
- Leverage `git mv` to preserve history and ease code review.
- Add unit/smoke tests alongside each extraction so reviewers can rely on CI instead of manual testing.

