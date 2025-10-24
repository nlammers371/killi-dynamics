"""Enhanced CZI export helpers with multi-side support.

This module extends :mod:`src.data_io.czi_export` to automatically detect and
export experiments that contain multiple fields of view ("sides").  Two common
acquisition patterns are handled:

* **List export** – each timepoint is stored as an individual CZI file.  For
  two-sided experiments each side uses a distinct filename prefix.  The code
  groups files by prefix and writes the result into separate Zarr groups.
* **ND export** – the entire time series is stored within a single CZI file
  containing an additional scene dimension.  Each scene is exported as its own
  Zarr group.

The resulting Zarr store contains one child array per detected side while
sharing a single top-level directory.
"""

from __future__ import annotations

import multiprocessing
import os
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import zarr
from bioio import BioImage
from skimage.transform import resize
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.registration._Archive.image_fusion import get_hemisphere_shifts
from src.utilities.functions import path_leaf

_CHUNK_KEY_RE = re.compile(r"^(\d+)\..*$")  # capture leading time index


@dataclass(frozen=True)
class SideSpec:
    """Description of a single side to export."""

    name: str
    image_list: List[Path]
    file_prefix: Optional[str]
    scene_index: Optional[int]
    source_type: str


def _existing_time_indices(zarr_dir: Path) -> set[int]:
    """Return a set of time indices that already have at least one chunk written."""

    seen: set[int] = set()
    try:
        for name in os.listdir(zarr_dir):
            match = _CHUNK_KEY_RE.match(name)
            if match:
                seen.add(int(match.group(1)))
    except FileNotFoundError:
        pass  # store not created yet
    return seen


def _extract_prefix(path: Path) -> str:
    """Infer the acquisition prefix for a CZI file."""

    prefix = re.sub(r"\(.*\)\.czi$", "", path.name, flags=re.IGNORECASE)
    if prefix == path.name:
        prefix = path.stem
    return prefix


def _detect_side_specs(image_files: List[Path]) -> List[SideSpec]:
    """Identify one or more :class:`SideSpec` instances from a list of files."""

    if not image_files:
        raise FileNotFoundError("No CZI files were found for export.")

    prefix_map: dict[str, List[Path]] = {}
    for file_path in image_files:
        prefix = _extract_prefix(file_path)
        prefix_map.setdefault(prefix, []).append(file_path)

    # Multiple prefixes imply a list-export two-sided acquisition
    if len(prefix_map) > 1:
        side_specs: List[SideSpec] = []
        for idx, (prefix, files) in enumerate(sorted(prefix_map.items())):
            files_sorted = sorted(files)
            side_specs.append(
                SideSpec(
                    name=f"side_{idx:02d}",
                    image_list=files_sorted,
                    file_prefix=prefix,
                    scene_index=None,
                    source_type="multi_prefix",
                )
            )
        return side_specs

    # Otherwise a single prefix exists.  Determine whether a multi-scene ND file
    # is present by inspecting the first image.
    first_image = image_files[0]
    im_object = BioImage(first_image)
    dims = im_object.dims
    dims_str = dims.order.upper()

    if len(image_files) == 1 and "S" in dims_str and dims["S"][0] > 1:
        side_specs = [
            SideSpec(
                name=f"side_{idx:02d}",
                image_list=[first_image],
                file_prefix=_extract_prefix(first_image),
                scene_index=idx,
                source_type="nd_scene",
            )
            for idx in range(dims["S"][0])
        ]
        return side_specs

    # Fallback: single side
    return [
        SideSpec(
            name="side_00",
            image_list=sorted(image_files),
            file_prefix=_extract_prefix(first_image),
            scene_index=None,
            source_type="single",
        )
    ]


def initialize_zarr_store(
    zarr_path: Path,
    image_list: List[Path],
    resampling_scale: Iterable[float],
    channel_to_keep: Optional[Iterable[bool]],
    overwrite_flag: bool = False,
    last_i: Optional[int] = None,
):
    """Initialize or reopen a Zarr array ready for writing."""

    resampling_scale = np.asarray(resampling_scale, dtype=float)
    if resampling_scale.shape != (3,):
        raise ValueError("resampling_scale must be an iterable of length 3 (Z, Y, X).")

    im_object = BioImage(image_list[0])
    dims = im_object.dims
    dims_str = dims.order.upper()
    channel_names = getattr(im_object, "channel_names", None)

    if "T" in dims_str and dims["T"][0] > 1:
        n_timepoints = dims["T"][0]
        time_stack_flag = True
        n_files = 1
    else:
        n_timepoints = len(image_list)
        time_stack_flag = False
        n_files = n_timepoints

    if last_i is not None:
        n_timepoints = min(n_timepoints, int(last_i))

    multichannel_flag = False
    n_channels = 1
    if "C" in dims_str:
        n_channels = dims["C"][0]
        if channel_to_keep is not None:
            keep = np.asarray(channel_to_keep, dtype=bool)
            n_channels = int(keep.sum())
        multichannel_flag = n_channels > 1

    space_shape = tuple(dims[k][0] for k in ["Z", "Y", "X"])

    raw_scale_vec = np.asarray(im_object.physical_pixel_sizes, dtype=float)
    if np.max(raw_scale_vec) <= 1e-5:
        raw_scale_vec *= 1e6

    if raw_scale_vec[0] != resampling_scale[0]:
        raise ValueError(
            "Z resampling not supported; input data Z spacing differs from target."
        )

    rs_factors = np.divide(raw_scale_vec, resampling_scale)
    out_spatial = tuple(np.round(np.multiply(space_shape, rs_factors)).astype(int))

    if multichannel_flag:
        inner_shape = (n_channels,) + out_spatial
        chunks = (1, 1) + inner_shape[1:]
    else:
        inner_shape = out_spatial
        chunks = (1,) + inner_shape

    shape_out = (n_timepoints,) + inner_shape
    dtype = np.uint16
    mode = "w" if overwrite_flag else "a"

    zarr_file = zarr.open(
        zarr_path,
        mode=mode,
        shape=shape_out,
        dtype=dtype,
        chunks=chunks,
    )

    if overwrite_flag:
        indices_to_write = list(range(n_timepoints))
    else:
        already = _existing_time_indices(zarr_path)
        already = {i for i in already if 0 <= i < n_timepoints}
        indices_to_write = sorted(set(range(n_timepoints)) - already)

    summary = dict(
        dims=dims,
        shape=shape_out,
        pixel_size_um=resampling_scale.tolist(),
        channels=channel_names,
        time_stack=time_stack_flag,
        n_files=n_files,
    )
    print(f"[initialize_zarr_store] Summary:\n{summary}")

    return zarr_file, indices_to_write, time_stack_flag


def write_zarr(
    t: int,
    zarr_file,
    image_list: List[Path],
    time_stack_flag: bool,
    file_prefix: Optional[str],
    resampling_scale,
    tres=None,
    channel_names=None,
    channels_to_keep=None,
    scene_index: Optional[int] = None,
    side_name: Optional[str] = None,
):
    """Write a single timepoint to the output Zarr array."""

    load_kwargs = {}
    if scene_index is not None:
        load_kwargs["S"] = scene_index

    if not time_stack_flag:
        im_path = Path(image_list[t])
        file_name = im_path.name
        time_point = t
        if file_prefix is not None:
            time_string = file_name.replace(file_prefix, "").replace(".czi", "")
            match = re.search(r"(\d+)", time_string)
            if match:
                try:
                    time_point = int(match.group(1)) - 1
                except ValueError:
                    time_point = t
        im_object = BioImage(im_path)
        arr = im_object.get_image_dask_data("CZYX", **load_kwargs)
    else:
        time_point = t
        im_path = Path(image_list[0])
        im_object = BioImage(im_path)
        arr = im_object.get_image_dask_data("CZYX", T=t, **load_kwargs)

    c_dim = arr.shape[0]
    if channels_to_keep is not None:
        keep = np.asarray(channels_to_keep, dtype=bool)
        arr = arr[keep, ...]
        c_dim = arr.shape[0]

    multichannel = arr.ndim == 4 and c_dim > 1
    if not multichannel:
        arr = np.squeeze(arr)

    target_shape = zarr_file.shape[1:]
    z_dim = target_shape[1] if multichannel else target_shape[0]
    yx_shape = target_shape[-2:]

    image_data_rs = np.empty(target_shape, dtype=np.uint16)

    for c in range(c_dim):
        channel_data = arr[c].compute()
        nz = channel_data.shape[0]
        resized_planes = np.zeros((z_dim, *yx_shape), dtype=np.uint16)
        for z in range(min(z_dim, nz)):
            resized_planes[z] = np.round(
                resize(
                    channel_data[z],
                    yx_shape,
                    preserve_range=True,
                    order=1,
                    anti_aliasing=True,
                )
            ).astype(np.uint16)

        if multichannel:
            image_data_rs[c] = resized_planes
        else:
            image_data_rs = resized_planes

    if t == 0:
        px_um = np.asarray(im_object.physical_pixel_sizes, dtype=float)
        if np.max(px_um) <= 1e-5:
            px_um *= 1e6
        pixel_size_um = tuple(px_um.tolist())

        if channel_names is None:
            if multichannel:
                channel_names = getattr(im_object, "channel_names", None)
                if channel_names is None:
                    channel_names = [f"channel{c:02d}" for c in range(c_dim)]
            else:
                channel_names = ["channel00"]

        metadata = {
            "dim_order": "TCZYX" if multichannel else "TZYX",
            "n_timepoints": int(zarr_file.shape[0]),
            "voxel_size_um": tuple(map(float, resampling_scale)),
            "time_resolution_s": float(tres) if tres is not None else None,
            "raw_voxel_scale_um": list(map(float, pixel_size_um)),
            "channels": list(map(str, channel_names)),
            "source_file": str(im_path),
            "scene_index": scene_index,
            "side_name": side_name,
        }
        zarr_file.attrs.update(metadata)
        print(f"[write_zarr] Metadata: {metadata}")

    zarr_file[time_point] = image_data_rs


def export_czi_to_zarr(
    raw_data_root: Path | str,
    project_name: str,
    tres: Optional[float] = None,
    save_root: Path | str | None = None,
    last_i: Optional[int] = None,
    overwrite_flag: bool = False,
    resampling_scale: Optional[Iterable[float]] = None,
    channel_names: Optional[List[str]] = None,
    channels_to_keep: Optional[Iterable[bool]] = None,
    n_workers: int = 8,
    file_prefix: str | Iterable[str] | None = None,
    *,
    register_two_sided: bool = True,
    registration_interval: int = 25,
    registration_nucleus_channel: int = 1,
    registration_z_align_size: int = 50,
    registration_start_i: int = 0,
    registration_last_i: Optional[int] = None,
    registration_n_workers: Optional[int] = None,
):
    """Export CZI datasets to Zarr with automatic multi-side handling.

    When a two-sided acquisition is detected, the hemisphere registration routine
    is executed automatically (unless ``register_two_sided`` is ``False``) and the
    resulting per-frame shifts are stored in the root Zarr group's attributes.
    """

    par_flag = n_workers > 1

    raw_data_root = Path(raw_data_root)
    if save_root is None:
        save_root = raw_data_root
    else:
        save_root = Path(save_root)

    if resampling_scale is None:
        resampling_scale = np.asarray([3, 0.85, 0.85])

    if channels_to_keep is not None:
        if channel_names is None:
            raise ValueError("channel_names must be provided if channels_to_keep is used.")
        channel_names = [ch for ch, keep in zip(channel_names, channels_to_keep) if keep]

    store_path = save_root / "built_data" / "zarr_image_files" / f"{project_name}.zarr"

    raw_path = raw_data_root / "raw_image_data" / project_name
    if not raw_path.exists():
        raise FileNotFoundError(f"Could not locate raw image directory: {raw_path}")

    if file_prefix is None:
        czi_files = sorted(raw_path.glob("*.czi"))
    else:
        prefixes = (
            [file_prefix]
            if isinstance(file_prefix, (str, Path))
            else list(file_prefix)
        )
        czi_files = []
        for prefix in prefixes:
            czi_files.extend(sorted(raw_path.glob(f"{prefix}*.czi")))
        czi_files = sorted(set(czi_files))

    side_specs = _detect_side_specs(czi_files)

    mode = "w" if overwrite_flag else "a"
    root_group = zarr.open_group(str(store_path), mode=mode)
    root_group.attrs.update(
        {
            "project_name": project_name,
            "sides": [
                {
                    "name": spec.name,
                    "file_prefix": spec.file_prefix,
                    "scene_index": spec.scene_index,
                    "source_type": spec.source_type,
                }
                for spec in side_specs
            ],
        }
    )

    side_zarr_paths: list[Path] = []

    for spec in side_specs:
        print(f"[export_czi_to_zarr_v2] Processing {spec.name} ({spec.source_type})")
        side_zarr_path = store_path / spec.name
        side_zarr_paths.append(side_zarr_path)
        zarr_file, indices_to_write, time_stack_flag = initialize_zarr_store(
            side_zarr_path,
            image_list=spec.image_list,
            resampling_scale=resampling_scale,
            channel_to_keep=channels_to_keep,
            overwrite_flag=overwrite_flag,
            last_i=last_i,
        )

        indices_to_write = np.asarray(indices_to_write)
        if last_i is not None:
            indices_to_write = indices_to_write[indices_to_write <= last_i]

        if not time_stack_flag:
            image_time_stamps: List[int] = []
            for image_path in spec.image_list:
                f_string = path_leaf(image_path)
                if spec.file_prefix is not None:
                    time_string = f_string.replace(spec.file_prefix, "").replace(".czi", "")
                    match = re.search(r"(\d+)", time_string)
                    if match:
                        image_time_stamps.append(int(match.group(1)) - 1)
                        continue
                image_time_stamps.append(len(image_time_stamps))
        else:
            image_time_stamps = list(range(zarr_file.shape[0]))

        image_indices = np.arange(len(image_time_stamps))
        indices_to_write = np.where(np.isin(image_indices, indices_to_write))[0]

        run_write_zarr = partial(
            write_zarr,
            zarr_file=zarr_file,
            image_list=spec.image_list,
            time_stack_flag=time_stack_flag,
            channel_names=channel_names,
            channels_to_keep=channels_to_keep,
            file_prefix=spec.file_prefix,
            tres=tres,
            resampling_scale=resampling_scale,
            scene_index=spec.scene_index,
            side_name=spec.name,
        )

        if len(indices_to_write) == 0:
            print(f"[export_czi_to_zarr_v2] No new data to write for {spec.name}.")
            continue

        if par_flag:
            process_map(run_write_zarr, indices_to_write, max_workers=n_workers, chunksize=1)
        else:
            for i in tqdm(indices_to_write, desc=f"Exporting {spec.name} to zarr"):
                run_write_zarr(i)

    if register_two_sided and len(side_specs) == 2:
        ref_spec, moving_spec = side_specs
        ref_path, moving_path = side_zarr_paths
        print(
            "[export_czi_to_zarr_v2] Running hemisphere registration between "
            f"{ref_spec.name} and {moving_spec.name}"
        )
        if registration_n_workers is None:
            total_cpus = multiprocessing.cpu_count()
            registration_workers_used = max(1, total_cpus // 4)
        else:
            registration_workers_used = registration_n_workers

        shift_df = get_hemisphere_shifts(
            root=str(save_root),
            side1_name=ref_spec.name,
            side2_name=moving_spec.name,
            interval=registration_interval,
            nucleus_channel=registration_nucleus_channel,
            z_align_size=registration_z_align_size,
            last_i=registration_last_i if registration_last_i is not None else last_i,
            start_i=registration_start_i,
            n_workers=registration_workers_used,
            side1_path=str(ref_path),
            side2_path=str(moving_path),
            csv_output_path=None,
        )

        registration_dict = {
            "reference_side": ref_spec.name,
            "moving_side": moving_spec.name,
            "parameters": {
                "interval": int(registration_interval),
                "nucleus_channel": int(registration_nucleus_channel),
                "z_align_size": int(registration_z_align_size),
                "start_i": int(registration_start_i),
                "last_i": (int(registration_last_i) if registration_last_i is not None else None),
                "requested_last_i": (int(last_i) if last_i is not None else None),
                "n_workers": int(registration_workers_used),
            },
            "shifts": {
                "frame": shift_df["frame"].astype(int).tolist(),
                "zs": shift_df["zs"].astype(float).tolist(),
                "ys": shift_df["ys"].astype(float).tolist(),
                "xs": shift_df["xs"].astype(float).tolist(),
            },
        }

        root_group.attrs["hemisphere_registration"] = registration_dict
        print("[export_czi_to_zarr] Stored hemisphere registration metadata in Zarr attributes")

    print("Done.")


__all__ = [
    "SideSpec",
    "export_czi_to_zarr",
    "initialize_zarr_store",
    "write_zarr",
]

