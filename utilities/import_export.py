from pathlib import Path
from typing import Optional, Tuple, Union
import warnings
import zarr
from zarr.storage import Store
import sqlalchemy as sqla
from ultrack.config.config import MainConfig
from ultrack.utils.array import create_zarr, large_chunk_size
import logging
from typing import Callable, Sequence, Tuple
from ultrack.core.database import NO_PARENT, NodeDB
import numpy as np
import pandas as pd
from toolz import curry
from ultrack.utils import array
from ultrack.config.dataconfig import DataConfig
from ultrack.utils.multiprocessing import multiprocessing_apply

LOG = logging.getLogger(__name__)

@curry
def _query_and_export_data_to_frame(
    time: int,
    image: np.array,
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

    export_func(time, image[time, :, :, :])

def export_segmentation_generic(
    data_config: DataConfig,
    image: np.array,
    export_func: Callable[[int, np.ndarray], None],
) -> None:
    """
    Generic function to run01_export segmentation masks, segments labeled by `track_id` from `df`.

    Parameters
    ----------
    data_config : DataConfig
        Data parameters configuration.
    df : pd.DataFrame
        Tracks dataframe indexed by node id.
    export_func : Callable[[int, np.ndarray], None]
        Export function, it receives as input a time index `t` and its respective uint16 labeled buffer.
    """

    # if "track_id" not in df.columns:
    #     raise ValueError(f"Dataframe must have `track_id` column. Found {df.columns}")

    # LOG.info(f"Exporting segmentation masks with {export_func}")

    shape = data_config.metadata["shape"]

    multiprocessing_apply(
        _query_and_export_data_to_frame(
            # database_path=data_config.database_path,
            image=image,
            export_func=export_func,
        ),
        sequence=range(shape[0]),
        n_workers=data_config.n_workers,
        desc="Exporting segmentation masks",
    )

def image_to_zarr(
        config: MainConfig,
        image: np.array,
        store_or_path: Union[None, Store, Path, str] = None,
        chunks: Optional[Tuple[int]] = None,
        overwrite: bool = False,
) -> zarr.Array:
    """
    Exports segmentations masks to zarr array, `track_df` assign the `track_id` to their respective segments.
    By changing the `store` this function can be used to write zarr arrays into disk.

    Parameters
    ----------
    config : MainConfig
        Configuration parameters.
    image : ndarray
        Tracks dataframe, must have `track_id` column and be indexed by node id.
    store_or_path : Union[None, Store, Path, str], optional
        Zarr storage or output path, if not provided zarr.TempStore is used.
    chunks : Optional[Tuple[int]], optional
        Chunk size, if not provided it chunks time with 1 and the spatial dimensions as big as possible.
    overwrite : bool, optional
        If True, overwrites existing zarr array.

    Returns
    -------
    zarr.Array
        Output zarr array.
    """

    shape = config.data_config.metadata["shape"]
    dtype = np.int32

    if isinstance(store_or_path, zarr.MemoryStore) and config.data_config.n_workers > 1:
        raise ValueError(
            "zarr.MemoryStore and multiple workers are not allowed. "
            f"Found {config.data_config.n_workers} workers in `data_config`."
        )

    if chunks is None:
        chunks = large_chunk_size(shape, dtype=dtype)

    if isinstance(store_or_path, Store):
        array = zarr.zeros(shape, dtype=dtype, store=store_or_path, chunks=chunks)

    else:
        array = create_zarr(
            shape,
            dtype=dtype,
            store_or_path=store_or_path,
            chunks=chunks,
            default_store_type=zarr.TempStore,
            overwrite=overwrite,
        )

    export_segmentation_generic(config.data_config, image, array.__setitem__)
    return array