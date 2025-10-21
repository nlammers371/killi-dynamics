"""Deprecated QC helpers retained for backwards compatibility."""
from warnings import warn

from src.qc.mask_qc import mask_qc_wrapper, perform_mask_qc
from src.qc.morphology import ellipsoid_axis_lengths

warn(
    "src.build_lightsheet.process_masks is deprecated; import from src.qc instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ellipsoid_axis_lengths", "mask_qc_wrapper", "perform_mask_qc"]
