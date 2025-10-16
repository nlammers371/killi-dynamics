"""Shadow-artifact detection helpers."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from skimage.measure import regionprops


def _sparse_projection(regions, image_shape: Tuple[int, int, int]) -> coo_matrix:
    """Construct a sparse matrix encoding XY projections of labeled regions."""
    Y, X = image_shape[1:]
    row_indices = []
    col_indices = []
    data = []

    for idx, prop in enumerate(regions):
        zmin, ymin, xmin, zmax, ymax, xmax = prop.bbox
        projection = np.any(prop.image, axis=0)
        if not projection.any():
            continue
        local_ys, local_xs = np.nonzero(projection)
        full_ys = local_ys + ymin
        full_xs = local_xs + xmin
        flat = full_ys * X + full_xs

        row_indices.extend([idx] * len(flat))
        col_indices.extend(flat.tolist())
        data.extend([1] * len(flat))

    return coo_matrix((data, (row_indices, col_indices)), shape=(len(regions), Y * X)).tocsr()


def filter_shadowed_labels(
    mask: np.ndarray,
    scale_vec: Iterable[float],
    z_prox_thresh: float,
    min_overlap: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove labels that appear shadowed by upstream objects."""
    props = regionprops(mask)
    if not props:
        return mask, np.array([], dtype=np.int32)

    centroids = np.array([p.centroid for p in props])
    proj_matrix = _sparse_projection(props, mask.shape)
    intersections = (proj_matrix @ proj_matrix.T).toarray()
    areas = proj_matrix.sum(axis=1).A1
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = np.divide(intersections, areas[:, None], where=areas[:, None] > 0)
    overlap[np.isnan(overlap)] = 0.0

    z_scale = float(scale_vec[0])
    z_dist = (centroids[:, 0][:, None] - centroids[:, 0][None, :]) * z_scale
    upstream = z_dist < 0
    proximal = z_dist >= z_prox_thresh
    candidate = upstream & proximal & (overlap >= min_overlap)

    shadowed = np.zeros(len(props), dtype=bool)
    for idx in range(len(props)):
        if shadowed[idx]:
            continue
        shadowed |= candidate[idx]

    keep_labels = np.array([p.label for i, p in enumerate(props) if not shadowed[i]], dtype=np.int32)
    filtered = np.where(np.isin(mask, keep_labels), mask, 0)
    return filtered.astype(mask.dtype, copy=False), keep_labels
