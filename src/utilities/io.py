import itertools
import logging
import shutil
import warnings
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union
from toolz import curry
import numpy as np
import zarr
from ultrack.utils.multiprocessing import multiprocessing_apply
from zarr.storage import Store

def validate_and_overwrite_path(
    path: Path, overwrite: bool, msg_type: Literal["cli", "api"]
) -> None:
    """Validates and errors existance of path (or dir) and overwrites it if requested."""

    if msg_type == "cli":
        msg = f"{path} already exists. Set `--overwrite` option to overwrite it."

    elif msg_type == "api":
        msg = f"{path} already exists. Set `overwrite=True` to overwrite it."

    else:
        raise ValueError(f"Invalid `msg_type` {msg_type}, must be `cli` or `api`.")

    if path.exists():
        if overwrite:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        else:
            raise ValueError(msg)

def large_chunk_size(
    shape: Tuple[int],
    dtype: Union[str, np.dtype],
    max_size: int = 2147483647,
) -> Tuple[int]:
    """
    Computes a large chunk size for a given `shape` and `dtype`.
    Large chunks improves the performance on Elastic Storage Systems (ESS).
    Leading dimension (time) will always be chunked as 1.

    Parameters
    ----------
    shape : Tuple[int]
        Input data shape.
    dtype : Union[str, np.dtype]
        Input data type.
    max_size : int, optional
        Reference maximum size, by default 2147483647

    Returns
    -------
    Tuple[int]
        Suggested chunk size.
    """
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    plane_shape = np.minimum(shape[-2:], 32768)

    if len(shape) == 3:
        chunks = (1, *plane_shape)
    elif len(shape) > 3:
        depth = min(max_size // (dtype.itemsize * np.prod(plane_shape)), shape[1])
        chunks = (1,) * (len(shape) - 3) + (depth, *plane_shape)
    else:
        raise NotImplementedError(
            f"Large chunk size only implemented for 3-or-more dimensional arrays. Found {len(shape) - 1}-dims."
        )

    return chunks

def create_zarr(
    shape: Tuple[int, ...],
    dtype: np.dtype,
    store_or_path: Union[Store, Path, str, None] = None,
    overwrite: bool = False,
    default_store_type: Store = zarr.MemoryStore,
    chunks: Optional[Tuple[int]] = None,
    **kwargs,
) -> zarr.Array:
    """Create a zarr array of zeros.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the array.
    dtype : np.dtype
        Data type of the array.
    store_or_path : Optional[Union[Path, str]], optional
        Path to store the array, if None a zarr.MemoryStore is used, by default None
    overwrite : bool, optional
        Overwrite existing file, by default False
    chunks : Optional[Tuple[int]], optional
        Chunk size, if not provided it chunks time with 1 and the spatial dimensions as big as possible.

    Returns
    -------
    zarr.Array
        Zarr array of zeros.
    """
    if "path" in kwargs:
        raise ValueError("`path` is not a valid argument, use `store_or_path` instead.")

    if store_or_path is None:
        store = default_store_type()

    elif isinstance(store_or_path, Store):
        store = store_or_path

    else:
        if isinstance(store_or_path, str):
            store_or_path = Path(store_or_path)

        validate_and_overwrite_path(store_or_path, overwrite, msg_type="api")

        store = zarr.NestedDirectoryStore(str(store_or_path))

    if chunks is None:
        chunks = large_chunk_size(shape, dtype=dtype)

    return zarr.zeros(shape, dtype=dtype, store=store, chunks=chunks, **kwargs)



@curry
def _query_and_export_data_to_frame(
    time: int,
    array,
    export_func: Callable[[int, np.ndarray], None],
) -> None:
    """Queries segmentation data from database and paints it according to their respective `df` `track_id` column.

    Parameters
    ----------
    time : int
        Frame time point to paint.
    database_path : str
        Database path.
    shape : Tuple[int]
        Frame shape.
    df : pd.DataFrame
        Tracks dataframe.
    export_func : Callable[[int, np.ndarray], None]
        Export function, it receives as input a time index `t` and its respective uint16 labeled buffer.
    """


    export_func(time, array)


def export_segmentation_generic(
    array: np.array,
    export_func: Callable[[int, np.ndarray], None],
    n_workers: int=2
) -> None:
    """
    Generic function to export segmentation masks, segments labeled by `track_id` from `df`.

    Parameters
    ----------
    data_config : DataConfig
        Data parameters configuration.
    df : pd.DataFrame
        Tracks dataframe indexed by node id.
    export_func : Callable[[int, np.ndarray], None]
        Export function, it receives as input a time index `t` and its respective uint16 labeled buffer.
    """

    shape = array.shape

    multiprocessing_apply(
        _query_and_export_data_to_frame(
            array=array,
            export_func=export_func,
        ),
        sequence=range(shape[0]),
        n_workers=n_workers,
        desc="Exporting segmentation masks",
    )
