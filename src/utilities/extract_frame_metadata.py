"""Compatibility layer for metadata helpers relocated to `src.data_io`."""
from warnings import warn

from src.data_io.nd2_metadata import (
    extract_frame_metadata,
    parse_curation_metadata,
    parse_nd2_metadata,
    parse_plate_metadata,
    permute_nd2_axes,
)

warn(
    "src.utilities.extract_frame_metadata is deprecated; "
    "import from src.data_io.nd2_metadata instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "extract_frame_metadata",
    "parse_curation_metadata",
    "parse_nd2_metadata",
    "parse_plate_metadata",
    "permute_nd2_axes",
]

