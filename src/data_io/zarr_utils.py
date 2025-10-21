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


def _side_aliases(index: int) -> list[str]:
    """Return a list of plausible group names for a given side index."""

    aliases = [
        f"side_{index:02d}",
        f"side{index:02d}",
        f"side_{index}",
        f"side{index}",
    ]

    if 0 <= index < 26:
        suffix = chr(ord("a") + index)
        aliases.extend([
            f"side_{suffix}",
            f"side{suffix}",
        ])

    return aliases


DEFAULT_SIDE_NAMES: tuple[str, ...] = tuple(
    name for idx in range(4) for name in _side_aliases(idx)
)


def _dedupe(sequence: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in sequence:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


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


def _parse_legacy_suffix(project_name: str) -> tuple[str, list[str]]:
    """Split a legacy ``*_sideX``/``*_fused`` name into base and aliases."""

    side_match = re.search(r"^(.*)_side[_-]?(\d+)$", project_name, flags=re.IGNORECASE)
    if side_match:
        base = side_match.group(1)
        index = int(side_match.group(2))
        indices: list[int] = []
        if index > 0:
            indices.append(index - 1)
        indices.append(index)
        aliases: list[str] = []
        for idx in indices:
            aliases.extend(_side_aliases(idx))
        return base, _dedupe(aliases)

    fused_match = re.search(r"^(.*)_(fused|fusion)$", project_name, flags=re.IGNORECASE)
    if fused_match:
        base = fused_match.group(1)
        return base, list(FUSED_ALIASES)

    return project_name, []


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


def open_experiment_array(
    root: Path | str,
    project_name: str,
    *,
    side: str | int | Sequence[str | int] | None = None,
    prefer_fused: bool = False,
) -> Tuple[zarr.Array, Path, Optional[str]]:
    """Open a project's image array, resolving legacy naming automatically."""

    root = Path(root)
    store_dir = root / "built_data" / "zarr_image_files"

    direct_path = store_dir / f"{project_name}.zarr"
    candidate_names = _expand_side_candidates(side)

    if direct_path.exists():
        array, group_name = open_image_array(direct_path, candidates=candidate_names, prefer_fused=prefer_fused)
        return array, direct_path, group_name

    base_name, legacy_candidates = _parse_legacy_suffix(project_name)
    store_path = store_dir / f"{base_name}.zarr"
    if not store_path.exists():
        raise FileNotFoundError(f"Could not locate a Zarr store for '{project_name}' at {store_path}")

    all_candidates = candidate_names + legacy_candidates
    array, group_name = open_image_array(store_path, candidates=all_candidates, prefer_fused=prefer_fused)
    return array, store_path, group_name


__all__ = [
    "open_image_array",
    "open_experiment_array",
]

