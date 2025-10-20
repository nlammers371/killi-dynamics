"""Legacy entrypoints for nuclei segmentation (deprecated)."""
from warnings import warn

from src.segmentation.cellpose import (
    cellpose_segmentation,
    segment_FOV,
    segment_fov,
)
from src.segmentation.mask_builders import (
    do_hierarchical_watershed,
    perform_li_segmentation,
)
from src.segmentation.postprocess import (
    estimate_li_thresh,
    segment_nuclei,
)
from src.segmentation.li_thresholding import (
    calculate_li_thresh,
    calculate_li_trend,
    extract_random_quadrant,
)

warn(
    "src.build_lightsheet.run02_segment_nuclei has moved; import from src.segmentation instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "calculate_li_thresh",
    "calculate_li_trend",
    "cellpose_segmentation",
    "do_hierarchical_watershed",
    "estimate_li_thresh",
    "extract_random_quadrant",
    "perform_li_segmentation",
    "segment_FOV",
    "segment_fov",
    "segment_nuclei",
]
