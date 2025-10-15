"""Segmentation primitives exposed under a stable namespace.

The actual implementations still live alongside the historical
lightsheet scripts; importing them here makes it possible to
progressively update callers without relocating the code yet.
"""
from src.build_lightsheet.run02_segment_nuclei import (
    calculate_li_trend,
    calculate_li_thresh,
    do_hierarchical_watershed,
    perform_li_segmentation,
)
from src.build_lightsheet.build_utils import labels_to_contours_nl

__all__ = [
    "calculate_li_trend",
    "calculate_li_thresh",
    "do_hierarchical_watershed",
    "perform_li_segmentation",
    "labels_to_contours_nl",
]
