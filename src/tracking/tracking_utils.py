import multiprocessing
from functools import partial
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import zarr
from numpy.typing import ArrayLike
from zarr.storage import Store
from ultrack.utils.array import create_zarr
from ultrack.utils.cuda import import_module, to_cpu
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from scipy import ndimage as ndi
import skimage.segmentation as segm

def labels_to_contours_nl(
    labels: Union[ArrayLike, Sequence[ArrayLike]],
    write_indices: Sequence[int],
    sigma: Optional[Union[Sequence[float], float]] = None,
    foreground_store_or_path: Union[Store, str, None] = None,
    contours_store_or_path: Union[Store, str, None] = None,
    overwrite: bool = False,
    n_workers: Optional[int] = None,
    par_flag: bool = True,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Converts and merges a sequence of labels into ultrack input format (foreground and contours)

    Parameters
    ----------
    labels : Union[ArrayLike, Sequence[ArrayLike]]
        List of labels with equal shape.
    sigma : Optional[Union[Sequence[float], float]], optional
        Contours smoothing parameter (gaussian blur), contours aren't smoothed when not provided.
    foreground_store_or_path : str, zarr.storage.Store, optional
        Zarr storage, it can be used with zarr.NestedDirectoryStorage to save the output into disk.
        By default it loads the data into memory.
    contours_store_or_path : str, zarr.storage.Store, optional
        Zarr storage, it can be used with zarr.NestedDirectoryStorage to save the output into disk.
        By default it loads the data into memory.
    overwrite : bool, optional
        Overwrite output output files if they already exist, by default False.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Combined foreground and edges arrays.
    """
    # ndi = import_module("scipy", "ndimage")
    # segm = import_module("skimage", "segmentation")

    # if not isinstance(labels, Sequence):
    #     labels = [labels]
    if isinstance(labels, Sequence):
        raise ValueError("Function is not yet compatible with multiple lablels per image")

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        # Limit yourself to 33% of CPUs (rounded down, at least 1)
        n_workers = max(1, total_cpus // 3)


    shape = (len(write_indices),) + labels.shape[1:]


    foreground = create_zarr(
        shape=shape,
        dtype=bool,
        store_or_path=foreground_store_or_path,
        overwrite=overwrite,
        default_store_type=zarr.TempStore,
    )
    contours = create_zarr(
        shape=shape,
        dtype=np.float32,
        store_or_path=contours_store_or_path,
        overwrite=overwrite,
        default_store_type=zarr.TempStore,
    )

    label_fun_run = partial(label_fun, write_indices=write_indices, labels=labels, foreground=foreground,
                            contours=contours, shape=shape, sigma=sigma)

    if par_flag:
        print("Using parallel processing")
        process_map(label_fun_run, write_indices, max_workers=n_workers, chunksize=1)
    else:
        print("Using sequential processing")
        for t in tqdm(write_indices):
            label_fun_run(t)

    return foreground, contours


def label_fun(t, write_indices, labels, foreground, contours, shape, sigma=None):

    foreground_frame = np.zeros(shape[1:], dtype=foreground.dtype)
    contours_frames = np.zeros(shape[1:], dtype=contours.dtype)

    lb_frame = np.asarray(labels[t])

    # get scale info
    # if scale_vec is None:
    #     scale_vec = (1.0, 1.0, 1.0)  # default scale if not provided
    #
    # if last_filter_start_i is not None and t >= last_filter_start_i:
    #     lb_frame = sh_mask_filter(lb_frame, scale_vec)

    foreground_frame |= lb_frame > 0
    contours_frames += segm.find_boundaries(lb_frame, mode="outer")

    contours_frames /= len(labels)

    if sigma is not None:
        contours_frames = ndi.gaussian_filter(contours_frames, sigma)
        contours_frames = contours_frames / contours_frames.max()

    out_index = np.where(write_indices == t)[0][0]  # find the index in write_indices
    foreground[out_index] = to_cpu(foreground_frame)
    contours[out_index] = to_cpu(contours_frames)


