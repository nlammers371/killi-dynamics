"""Utilities for resolving experiment image Zarr stores.

These helpers centralise the logic required to cope with historical
export layouts (one store per side) as well as the refactored layout where
multiple sides live as child groups inside a single store.  Downstream code
can simply request an experiment/side and rely on the helpers to pick the
appropriate dataset, falling back to the first available array when the
requested target is absent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re
from pathlib import Path
from typing import Optional, Tuple

import zarr

FUSED_ALIASES: tuple[str, ...] = ("fused", "fusion", "fused_image")

import zarr
from pathlib import Path
from typing import Optional, Any, Dict, Literal
from src.registration.virtual_fusion import VirtualFuseArray
from src.export.legacy_helpers import (_parse_legacy_suffix, DEFAULT_SIDE_NAMES, _side_aliases, _dedupe)
import logging
logger = logging.getLogger(__name__)  # <-- add this near the top of the module

# def get_metadata(
#     root: Path | str,
#     project_name: str,
#     *,
#     group_name: Optional[str] = None,
# ) -> Dict[str, Any]:
#     """
#     Return the attribute dictionary from the first array or group
#     inside a Zarr store.
#
#     Parameters
#     ----------
#     store_path : Path or str
#         Path to the Zarr store directory.
#     group_name : str, optional
#         If provided, return attrs from this subgroup instead of the first found.
#
#     Returns
#     -------
#     attrs : dict
#         The attribute dictionary (may be empty if no attrs found).
#     """
#     root = Path(root)
#     store_dir = root / "built_data" / "zarr_image_files"
#     store_path = store_dir / f"{project_name}.zarr"
#     store = zarr.open(store_path, mode="r")
#
#     # Direct array case
#     if isinstance(store, zarr.Array):
#         return dict(store.attrs)
#
#     # If a specific subgroup is requested
#     if group_name is not None and group_name in store:
#         node = store[group_name]
#         return dict(node.attrs)
#
#     # Otherwise, use your existing first-array traversal logic
#     for name, array in store.arrays():
#         return dict(array.attrs)
#     for name, subgroup in store.groups():
#         attrs = get_first_attrs(store_path / name)
#         if attrs:
#             return attrs
#
#     # If nothing found, return empty dict
#     return {}



def _first_array(node: zarr.Group | zarr.Array) -> Optional[zarr.Array]:
    """Return the first array reachable from ``node`` (depth-first)."""

    if isinstance(node, zarr.Array):
        return node

    for _name, array in node.arrays():
        return array

    for _name, subgroup in node.groups():
        array = _first_array(subgroup)
        if array is not None:
            return array

    return None


def _expand_side_candidates(side: str | int | Sequence[str | int]) -> list[str]:
    """Normalise side specifiers into candidate group names."""

    if side is None:
        return []

    if isinstance(side, (str, int)):
        values: Sequence[str | int] = [side]
    else:
        values = side

    candidates: list[str] = []
    for value in values:
        if isinstance(value, int):
            candidates.extend(_side_aliases(value))
            continue

        value_str = value.strip("/")
        candidates.append(value_str)

        match = re.search(r"(\d+)$", value_str)
        if match:
            index = int(match.group(1))
            # The legacy layout used 1-based numbering; try both.
            if index > 0:
                candidates.extend(_side_aliases(index - 1))
            candidates.extend(_side_aliases(index))

    return _dedupe(candidates)


def open_image_array(
    store_path: Path | str,
    *,
    candidates: Sequence[str] | None = None,
    prefer_fused: bool = False,
) -> Tuple[zarr.Array, Optional[str]]:
    """Open an image Zarr array from ``store_path``.

    Parameters
    ----------
    store_path:
        Path to the Zarr store directory.
    candidates:
        Optional ordered iterable of group names to try.
    prefer_fused:
        If ``True`` the helper will try fused datasets before falling back to
        side-specific groups.

    Returns
    -------
    array, group_name:
        The resolved array and the group key (``None`` if the array lives at
        the store root).
    """

    path = Path(store_path)
    node = zarr.open(path.as_posix(), mode="r")

    if isinstance(node, zarr.Array):
        return node, None

    candidate_names: list[str] = []

    fused_options = list(FUSED_ALIASES)
    if prefer_fused:
        candidate_names.extend(fused_options)

    if candidates:
        candidate_names.extend(candidates)
    else:
        candidate_names.extend(DEFAULT_SIDE_NAMES)

    if not prefer_fused:
        candidate_names.extend(fused_options)

    candidate_names = _dedupe(candidate_names)

    for name in candidate_names:
        if name not in node:
            continue
        array = _first_array(node[name])
        if array is not None:
            return array, name

    array = _first_array(node)
    if array is not None:
        return array, None

    raise KeyError(
        f"Could not locate an image array inside {store_path}. "
        "Checked candidates: " + ", ".join(candidate_names)
    )


def _gpu_available() -> bool:
    """Return True if a working CUDA GPU is available via CuPy."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def open_experiment_array(
    root: Path | str,
    project_name: str,
    *,
    side: str | None = None,
    prefer_fused: bool = True,
    use_gpu: bool | None = None,
    well_num: int | None = None,
    verbose: bool = False,
    mode: str = "r",
) -> Tuple[zarr.Array | VirtualFuseArray, Path, Optional[str]]:
    """
    Open a project's image array, resolving legacy naming automatically.

    Extended version:
      • Automatically detects and loads fused or two-sided experiments.
      • Uses GPU-backed VirtualFuseArray when available.
      • Backward-compatible: returns same (array, store_path, group_name) tuple.

    Parameters
    ----------
    root : Path | str
        Root project directory containing built_data/zarr_image_files.
    project_name : str
        Base name of the dataset (without .zarr).
    side : str | int | Sequence[str | int] | None
        Side selection override (legacy).
    prefer_fused : bool
        If True, prefer a persistent fused group if it exists.
    use_gpu : bool | None
        Whether to enable GPU acceleration. If None, autodetect.
    mode : str
        Open mode for zarr.open_group (default: "r").

    Returns
    -------
    array : zarr.Array | VirtualFuseArray
    store_path : Path
    group_name : Optional[str]
    """

    # get path to store
    root = Path(root)
    image_dir = root / "built_data" / "zarr_image_files"
    if well_num is not None:
        store_dir = image_dir / f"{project_name}_well{well_num:04}.zarr"
    else:
        store_dir = image_dir / f"{project_name}.zarr"
    root_group = zarr.open_group(store_dir, mode=mode)
    side_keys = sorted([k for k in root_group.array_keys() if k.startswith("side_")])

    if use_gpu is None:
        use_gpu = _gpu_available()

    # if a specific side is requested, return it
    if side is not None:
        if side in side_keys:
            za = root_group[side]
        elif side == "virtual_fused" or side == "fused":
            interp = "linear" if use_gpu else "nearest"
            if verbose:
                print(
                    f"[open_experiment_array] Two-sided experiment detected → "
                    f"VirtualFuseArray(interp='{interp}', backend={'GPU' if use_gpu else 'CPU'})"
                )
            za = VirtualFuseArray(store_dir, use_gpu=use_gpu, interp=interp)
        return za, store_dir, side


    # --- 4️⃣ Try persistent fused group first ---
    if prefer_fused and "fused" in root_group:
        if verbose:
            print(f"[open_experiment_array] Using persistent fused array at {store_dir}/fused")
        return root_group["fused"], store_dir, "fused"

    # --- 5️⃣ Detect two-sided structure ---
    if {"side_00", "side_01"}.issubset(side_keys):
        interp = "linear" if use_gpu else "nearest"
        if verbose:
            print(
                f"[open_experiment_array] Two-sided experiment detected → "
                f"VirtualFuseArray(interp='{interp}', backend={'GPU' if use_gpu else 'CPU'})"
            )
        vf = VirtualFuseArray(store_dir, use_gpu=use_gpu, interp=interp)
        return vf, store_dir, "virtual_fused"

    # --- 6️⃣ Otherwise fall back to standard side loading ---
    array = root_group[side_keys[0]]
    if verbose:
        print(
            f"[open_experiment_array] Loading first available side '{side_keys[0]}' from {store_dir}"
        )

    return array, store_dir, side_keys[0]



def open_mask_array(
    root: Path | str,
    project_name: str,
    seg_type: str = "li_segmentation",
    *,
    side: str | None = None,
    verbose: bool = False,
    prefer_fused: bool = True,
    mask_field: str = "clean",
    well_num: int | None = None,
    use_gpu: bool | None = None,
    mode: str = "r",
) -> Tuple[zarr.Array | VirtualFuseArray, Path, Optional[str]]:

    # get path to store
    root = Path(root)
    mask_dir = root / "segmentation" / seg_type
    if well_num is not None:
        store_dir = mask_dir / f"{project_name}_well{well_num:04}_masks.zarr"
    else:
        store_dir = mask_dir / f"{project_name}_masks.zarr"

    root_group = zarr.open(store_dir, mode=mode)
    side_keys = sorted([k["name"] for k in root_group.attrs["sides"]])

    if use_gpu is None:
        use_gpu = _gpu_available()

    # if a specific side is requested, return it
    if side is not None:
        if side in side_keys:
            za = root_group[side][mask_field]
        elif side == "virtual_fused":
            if verbose:
                logger.info(
                    f"[open_mask_array] Two-sided experiment detected → "
                    f"VirtualFuseArray(interp='nearest', backend={'GPU' if use_gpu else 'CPU'})"
                )
            za = VirtualFuseArray(store_dir, use_gpu=use_gpu, interp="nearest", is_mask=True, subgroup_key=mask_field)
        else:
            raise KeyError(f"Requested side '{side}' not found in mask store at {store_dir}.")
        return za, store_dir, side

    # --- 4️⃣ Try persistent fused group first ---
    if prefer_fused and "fused" in root_group:
        if verbose:
            logger.info(f"[open_mask_array] Using persistent fused array at {store_dir}/fused")
        return root_group["fused"][mask_field], store_dir, "fused"

    # --- 5️⃣ Detect two-sided structure ---
    if {"side_00", "side_01"}.issubset(side_keys):
        if verbose:
            logger.info(
                f"[open_mask_array] Two-sided experiment detected → "
                f"VirtualFuseArray(interp='nearest', backend={'GPU' if use_gpu else 'CPU'})"
            )
        vf = VirtualFuseArray(store_dir, use_gpu=use_gpu, interp="nearest", is_mask=True, subgroup_key=mask_field)
        return vf, store_dir, "virtual_fused"

    # --- 6️⃣ Otherwise fall back to standard side loading ---
    array = root_group[side_keys[0]][mask_field]
    if verbose:
        logger.info(f"[open_mask_array] Loading first available side '{side_keys[0]}' from {store_dir}")

    return array, store_dir, side_keys[0]



__all__ = [
    "open_image_array",
    "open_experiment_array",
]

