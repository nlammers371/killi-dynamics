"""Backward-compatible ND2 export shim."""
from warnings import warn

from src.data_io.nd2_export import export_nd2_to_zarr, write_to_zarr
from src.data_io.nd2_metadata import (
    extract_frame_metadata,
    parse_curation_metadata,
    parse_nd2_metadata,
    parse_plate_metadata,
    permute_nd2_axes,
)

warn(
    "src.build_yx1.export_nd2_to_zarr is deprecated; "
    "import from src.data_io.nd2_export instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "export_nd2_to_zarr",
    "write_to_zarr",
    "extract_frame_metadata",
    "parse_curation_metadata",
    "parse_nd2_metadata",
    "parse_plate_metadata",
    "permute_nd2_axes",
]

