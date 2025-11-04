import re
import shutil
from pathlib import Path
from typing import Optional, Literal, Iterable
import zarr
import numpy as np


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




def merge_zarr_sides(
    side1_path: Path | str,
    side2_path: Path | str,
    out_store_path: Path | str,
    project_name: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """
    Combine two legacy per-side Zarr stores into a single multi-side store
    matching the layout expected by `export_czi_to_zarr()`.

    Parameters
    ----------
    side1_path : Path or str
        Path to the first side Zarr store (e.g. 'side1.zarr').
    side2_path : Path or str
        Path to the second side Zarr store (e.g. 'side2.zarr').
    out_store_path : Path or str
        Destination for the unified multi-side store (e.g. 'myproject.zarr').
    project_name : str, optional
        Name to record in the root Zarr attributes. If None, inferred from out_store_path.
    overwrite : bool, default=False
        Whether to overwrite an existing destination.

    Returns
    -------
    Path
        Path to the new combined store.
    """
    side1_path = Path(side1_path)
    side2_path = Path(side2_path)
    out_store_path = Path(out_store_path)

    if out_store_path.exists():
        if not overwrite:
            raise FileExistsError(f"{out_store_path} already exists. Use overwrite=True to replace it.")
        shutil.rmtree(out_store_path)

    # Create parent group
    root = zarr.open_group(str(out_store_path), mode="w")

    # Copy side directories into subgroups
    side_map = [("side_00", side1_path), ("side_01", side2_path)]
    for side_name, src in side_map:
        dst = out_store_path / side_name
        shutil.move(src, dst)
        print(f"✅ Copied {src} → {dst}")

    # Fill in minimal metadata at the root
    if project_name is None:
        project_name = out_store_path.stem

    sides_metadata = [
        {"name": name, "file_prefix": None, "scene_index": None, "source_type": "legacy"}
        for name, _ in side_map
    ]

    root.attrs.update({
        "project_name": project_name,
        "sides": sides_metadata,
    })

    # check and update metadata as needed
    for side_name, _ in side_map:
        attrs = dict(root[side_name].attrs)

        # Convert voxel sizes
        if all(k in attrs for k in ("PhysicalSizeZ", "PhysicalSizeY", "PhysicalSizeX")):
            try:
                voxel_tuple = (
                    float(attrs["PhysicalSizeZ"]),
                    float(attrs["PhysicalSizeY"]),
                    float(attrs["PhysicalSizeX"]),
                )
                attrs["voxel_size_um"] = voxel_tuple
            except (ValueError, TypeError):
                pass  # skip malformed entries

        # Rename and normalize legacy fields
        if "TimeRes" in attrs:
            attrs["time_resolution_s"] = float(attrs.pop("TimeRes"))
        if "Channels" in attrs:
            attrs["channels"] = [str(ch) for ch in attrs.pop("Channels")]
        if "DimOrder" in attrs:
            attrs["dim_order"] = str(attrs.pop("DimOrder"))

        # Ensure required fields
        attrs.setdefault("n_time_points", int(root[side_name].shape[0]))
        attrs["side_name"] = side_name
        for key in ("raw_voxel_scale_um", "source_file", "scene_index"):
            attrs.setdefault(key, None)

        # Commit changes
        root[side_name].attrs.update(attrs)


    print(f"✅ Created unified multi-side store: {out_store_path}")
    return out_store_path


def convert_shift_table_to_transform_attrs(
    fused_store_path: Path | str,
    shift_csv_path: Path | str,
    moving_side: Literal["side_00", "side_01"] = "side_01",
    reference_side: Literal["side_00", "side_01"] = "side_00",
    flips_moving=(True, False, True),
    overwrite_existing: bool = True,
) -> None:
    """
    Convert a legacy per-frame shift CSV into new-style rigid_transform metadata,
    saving the shift table in a dedicated 'registration' directory.

    Layout:
        project_fused.zarr/
        ├── side_00/
        ├── side_01/
        ├── registration/
        │     ├── shifts_side_01.csv
        └── .zattrs

    Parameters
    ----------
    fused_store_path : Path or str
        Path to the unified multi-side Zarr store.
    shift_csv_path : Path or str
        CSV with columns ['frame','zs','ys','xs'] from registration.
    moving_side : str
        Name of the moving side (default: 'side_01').
    reference_side : str
        Name of the fixed reference side (default: 'side_00').
    flips_moving : tuple(bool,bool,bool)
        Axis-wise flips for the moving side.
    overwrite_existing : bool
        Overwrite existing rigid_transform attributes if present.
    """
    fused_store_path = Path(fused_store_path)
    shift_csv_path = Path(shift_csv_path)

    # Load the store
    root = zarr.open_group(fused_store_path, mode="a")

    # Load shift data
    df = pd.read_csv(shift_csv_path)
    required_cols = {"frame", "zs", "ys", "xs"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Shift CSV must contain columns {required_cols}")

    # --- Create registration subdir and write shift table there ---
    reg_dir = fused_store_path / "registration"
    reg_dir.mkdir(exist_ok=True)

    reg_csv_path = reg_dir / f"shifts_{moving_side}.csv"
    df.to_csv(reg_csv_path, index=False)
    print(f"✅ Saved per-frame shift table → {reg_csv_path}")

    # --- Rotation matrices ---
    R_ref = np.eye(3)
    R_mov = np.diag([(-1 if f else 1) for f in flips_moving])

    # --- Attribute dictionaries ---
    ref_attrs = {
        "reference_frame": "fused",
        "rotation": R_ref.tolist(),
        "flip": [False, False, False],
        "per_frame_shifts": None,
    }

    mov_attrs = {
        "reference_frame": "fused",
        "rotation": R_mov.tolist(),
        "flip": list(flips_moving),
        "per_frame_shifts": str(reg_csv_path.relative_to(fused_store_path)),
    }

    # --- Write back to Zarr attrs ---
    if overwrite_existing or "rigid_transform" not in root[reference_side].attrs:
        root[reference_side].attrs["rigid_transform"] = ref_attrs
    if overwrite_existing or "rigid_transform" not in root[moving_side].attrs:
        root[moving_side].attrs["rigid_transform"] = mov_attrs

    print(f"✅ Updated rigid_transform attrs for {reference_side} and {moving_side}")