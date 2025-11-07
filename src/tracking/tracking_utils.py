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

from pathlib import Path
import numpy as np
import multiprocessing
from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import zarr
from typing import Union, Sequence, Optional, Tuple
from numpy.typing import ArrayLike


def labels_to_contours_nl(
    labels: ArrayLike,
    write_indices: Sequence[int],
    sigma: Optional[Union[Sequence[float], float]] = None,
    foreground_store_or_path: Union[zarr.storage.Store, str, None] = None,
    contours_store_or_path: Union[zarr.storage.Store, str, None] = None,
    n_workers: Optional[int] = None,
    par_flag: bool = True,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Converts a sequence of label images into ultrack input format (foreground + contours).
    Each parallel worker writes its slice directly into the given Zarr datasets.
    """
    if isinstance(labels, Sequence):
        raise ValueError("Function is not yet compatible with multiple label stacks per call.")

    if n_workers is None:
        total_cpus = multiprocessing.cpu_count()
        n_workers = max(1, total_cpus // 3)

    # --- open existing datasets (no shape/dtype declared) ---
    if foreground_store_or_path is None or contours_store_or_path is None:
        raise ValueError("Must provide valid foreground and contours Zarr destinations.")

    foreground = zarr.open(foreground_store_or_path, mode="a")
    contours = zarr.open(contours_store_or_path, mode="a")

    shape = foreground.shape  # for reference if needed downstream

    # --- per-frame function ---
    run_frame = partial(
        label_fun,
        write_indices=write_indices,
        labels=labels,
        foreground=foreground,
        contours=contours,
        shape=shape,
        sigma=sigma,
    )

    # --- run computation ---
    if par_flag:
        print(f"Using {n_workers} workers for segmentation + direct writes")
        process_map(run_frame, write_indices, max_workers=n_workers, chunksize=1)
    else:
        print("Using sequential processing")
        for t in tqdm(write_indices, desc="Segmenting + writing"):
            run_frame(t)

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


