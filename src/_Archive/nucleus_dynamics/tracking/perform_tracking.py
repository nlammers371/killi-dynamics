"""Legacy tracking entry points (deprecated)."""
from warnings import warn

from src.tracking.workflow import (
    check_tracking,
    combine_tracking_results,
    copy_zarr,
    perform_tracking,
    reindex_mask,
    write_dask_to_zarr,
)

warn(
    "src.nucleus_dynamics.tracking.perform_tracking is deprecated; import from src.tracking instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "check_tracking",
    "combine_tracking_results",
    "copy_zarr",
    "perform_tracking",
    "reindex_mask",
    "write_dask_to_zarr",
]
