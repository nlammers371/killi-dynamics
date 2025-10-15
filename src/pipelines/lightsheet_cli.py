"""Example CLI orchestrating the lightsheet pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.qc import mask_qc_wrapper
from src.tracking import perform_tracking
from src.geometry import sphere_fit_wrapper


def run_lightsheet_pipeline(
    root: Path,
    project: str,
    seg_model: str,
    tracking_config: str,
    start: int,
    stop: int | None,
    execute: bool = False,
) -> None:
    """Run (or dry-run) QC, tracking, and sphere fitting for a project."""
    print("=== Lightsheet pipeline ===")
    print(f"Root: {root}")
    print(f"Project: {project}")
    print(f"Segmentation model: {seg_model}")
    print(f"Tracking config: {tracking_config}")
    if stop is None:
        print("Frame range: %s → end" % start)
    else:
        print(f"Frame range: {start} → {stop}")

    if not execute:
        print("Dry run: no actions executed. Use --execute to run the steps.")
        return

    print("\n[1/3] Running mask QC...")
    mask_qc_wrapper(root.as_posix(), project)

    print("\n[2/3] Launching Ultrack...")
    perform_tracking(
        root.as_posix(),
        project,
        tracking_config=tracking_config,
        seg_model=seg_model,
        start_i=start,
        stop_i=stop,
        use_fused=True,
    )

    print("\n[3/3] Fitting embryo spheres...")
    sphere_fit_wrapper(root, project, seg_model)
    print("Pipeline completed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the lightsheet processing pipeline.")
    parser.add_argument("root", type=Path, help="Repository or data root containing built_data/")
    parser.add_argument("project", help="Project name (used for zarr directories)")
    parser.add_argument("seg_model", help="Segmentation model directory name")
    parser.add_argument("tracking_config", help="Ultrack config name (without extension)")
    parser.add_argument("--start", type=int, default=0, help="First frame index for tracking")
    parser.add_argument("--stop", type=int, default=None, help="Optional last frame index")
    parser.add_argument("--execute", action="store_true", help="Execute instead of dry run")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_lightsheet_pipeline(
        root=args.root,
        project=args.project,
        seg_model=args.seg_model,
        tracking_config=args.tracking_config,
        start=args.start,
        stop=args.stop,
        execute=args.execute,
    )


if __name__ == "__main__":
    main()
