"""Backward-compatible shims for Cellpose segmentation."""

from warnings import warn

from src.segmentation.cellpose import cellpose_segmentation, segment_FOV, segment_fov

warn(
    "src.nucleus_dynamics.build.build01_segment_nuclei_zarr has moved; import from src.segmentation.cellpose instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["cellpose_segmentation", "segment_FOV", "segment_fov"]

