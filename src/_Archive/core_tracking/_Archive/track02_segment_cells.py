"""
Image segmentation via Cellpose library
"""
# from tifffile import TiffWriter
from tqdm import tqdm
import logging
import os
import time
import skimage.io as io
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
import numpy as np
from cellpose import models
from cellpose.core import use_gpu
from skimage.transform import resize
from src.utilities.image_utils import calculate_LoG
from src.utilities.functions import path_leaf
import json
import zarr
from src.utilities.image_utils import process_raw_image
# logging = logging.getlogging(__name__)
logging.basicConfig(level=logging.NOTSET)


# __OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def segment_FOV(
        column: np.ndarray,
        model=None,
        do_3D: bool = True,
        anisotropy=None,
        diameter: float = 30,
        cellprob_threshold: float = -8,
        flow_threshold: float = 0.4,
        min_size=None,
        label_dtype=None,
        pretrain_flag=False,
        return_probs=False
):
    """
    Internal function that runs Cellpose segmentation for a single ROI.

    :param column: Three-dimensional numpy array
    :param model: TBD
    :param do_3D: TBD
    :param anisotropy: TBD
    :param diameter: TBD
    :param cellprob_threshold: TBD
    :param flow_threshold: TBD
    :param min_size: TBD
    :param label_dtype: TBD
    """

    # Write some debugging info
    logging.info(
        f"[segment_FOV] START Cellpose |"
        f" column: {type(column)}, {column.shape} |"
        f" do_3D: {do_3D} |"
        f" model.diam_mean: {model.diam_mean} |"
        f" diameter: {diameter} |"
        f" flow threshold: {flow_threshold}"
    )

    # Actual labeling
    t0 = time.perf_counter()
    if not pretrain_flag:
        mask, flows, styles = model.eval(
            column,
            channels=[0, 0],
            do_3D=do_3D,
            net_avg=False,
            augment=False,
            diameter=diameter,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
        )
    else:
        if do_3D:
            mask, flows, styles = model.eval(
                column,
                channels=[0, 0],
                do_3D=do_3D,
                min_size=min_size,
                diameter=diameter,
                anisotropy=anisotropy,
                cellprob_threshold=cellprob_threshold,
                net_avg=False,
                augment=False
            )
        else:
            mask, flows, styles = model.eval(
                column,
                channels=[0, 0],
                do_3D=do_3D,
                min_size=min_size,
                diameter=diameter,
                anisotropy=anisotropy,
                stitch_threshold=0.3,
                cellprob_threshold=cellprob_threshold,
                net_avg=False,
                augment=False
            )
    # if not do_3D:
    #     mask = np.expand_dims(mask, axis=0)
    t1 = time.perf_counter()

    # Write some debugging info
    logging.info(
        f"[segment_FOV] END   Cellpose |"
        f" Elapsed: {t1 - t0:.4f} seconds |"
        f" mask shape: {mask.shape},"
        f" mask dtype: {mask.dtype} (before recast to {label_dtype}),"
        f" max(mask): {np.max(mask)} |"
        f" model.diam_mean: {model.diam_mean} |"
        f" diameter: {diameter} |"
        f" anisotropy: {anisotropy} |"
        f" flow threshold: {flow_threshold}"
    )
    if return_probs:
        probs = flows[2]
    else:
        probs = []

    return mask.astype(label_dtype), probs


def cellpose_segmentation(
        *,
        # Fractal arguments
        root: str,
        project_name: str,
        # Task-specific arguments
        seg_channel_label: Optional[str] = None,
        cell_diameter: float = 15,
        cellprob_threshold: float = -8,
        flow_threshold: float = 0.4,
        model_type: Literal["nuclei", "cyto", "cyto2"] = "nuclei",
        pretrained_model: Optional[str] = None,
        overwrite: Optional[bool] = False,
        return_probs: Optional[bool] = False,
        ds_factor: Optional[float] = 1.0,
        adjust_background: Optional[bool] = False
        # tiff_stack_mode = False,
        # pixel_res_input = None
) -> Dict[str, Any]:
    """
    Run cellpose segmentation on the ROIs of a single OME-NGFF image

    Full documentation for all arguments is still TBD, especially because some
    of them are standard arguments for Fractal tasks that should be documented
    in a standard way. Here are some examples of valid arguments::

        input_paths = ["/some/path/*.zarr"]
        component = "some_plate.zarr/B/03/0"
        metadata = {"num_levels": 4, "coarsening_xy": 2}

    :param raw_directory: path to directory containing zarr folders for images to run02_segment
    :param seg_channel_label: Identifier of a channel based on its label (e.g.
                          ``DAPI``). If not ``None``, then ``wavelength_id``
                          must be ``None``.
    :param cell_diameter: Initial diameter to be passed to
                            ``CellposeModel.eval`` method (after rescaling from
                            full-resolution to ``level``).
    :param output_label_name: output name for labels
    :param cellprob_threshold: Parameter of ``CellposeModel.eval`` method.
    :param flow_threshold: Parameter of ``CellposeModel.eval`` method.
    :param model_type: Parameter of ``CellposeModel`` class.
    :param pretrained_model: Parameter of ``CellposeModel`` class (takes
                             precedence over ``model_type``).
    """

    # Read useful parameters from metadata
    min_size = 1  # let's be maximally conservative here     # (cell_diameter/4)**3 / xy_ds_factor**2
    model_name = path_leaf(pretrained_model)
    # if tiff_stack_mode:
    #     if pixel_res_input is None:
    #         raise Exception("User must input pixel resolutions if using tiff stack mode")

    # raw_directory = os.path.join(root, "built_data", project_name, '')
    # if tiff

    # if not os.path.isdir(save_directory):
    #     os.makedirs(save_directory)


    # get list of images
    zarr_path = os.path.join(root, "built_data", "exported_image_files", project_name + ".zarr")
    data_tzyx = zarr.open(zarr_path, mode='r')

    metadata_file_path = os.path.join(root, "metadata", project_name, "metadata.json")
    f = open(metadata_file_path)

    # returns JSON object as
    # a dictionary
    metadata = json.load(f)

    pixel_res_raw = np.asarray([metadata["PhysicalSizeZ"], metadata["PhysicalSizeY"], metadata["PhysicalSizeX"]])
    metadata["ProbPhysicalSizeZ"] = pixel_res_raw[0] * ds_factor
    metadata["ProbPhysicalSizeY"] = pixel_res_raw[1] * ds_factor
    metadata["ProbPhysicalSizeX"] = pixel_res_raw[2] * ds_factor

    # anisotropy = pixel_res_raw[0] / pixel_res_raw[1]

    label_zarr_name = os.path.join(root, "built_data", "cellpose_output", model_name, project_name + "_labels.zarr")
    label_zarr = zarr.open(label_zarr_name, mode="a", shape=data_tzyx.shape, dtype=np.uint16, chunks=data_tzyx.shape)
    if return_probs:
        prob_zarr_name = os.path.join(root, "built_data", "cellpose_output", model_name,
                                                project_name + "_probs.zarr")
        prob_zarr = zarr.open(prob_zarr_name, mode="a", shape=data_tzyx.shape, dtype=np.float64,
                               chunks=data_tzyx.shape)
    # if not os.path.isdir(save_directory):
    #     os.makedirs(save_directory)

    iter_i = 0
    for im in tqdm(range(data_tzyx.shape[0])): #range(len(image_list[0] + image_list[575]))):
        # image_path = image_list[im]
        # im_name = path_leaf(image_path).replace(".tiff", "")
        # get time index
        # t = int(im_name[-4:])

        # label_name = im_name + f"_t{im:03}_labels"
        # label_path = os.path.join(save_directory, label_name)
        # if (not os.path.isfile(label_path + '.tif')) | overwrite:
        #     pass
        segment_flag = True
        if np.any(label_zarr[im, :, :, :] != 0) and (overwrite == False):
            segment_flag = False

        if segment_flag:
            # print("processing " + im_name)
            # read the image data

            data_zyx_raw = np.squeeze(data_tzyx[im, :, :, :])
            anisotropy_raw = pixel_res_raw[0] / pixel_res_raw[1]
            # rescale data
            dims_orig = data_zyx_raw.shape

            if ds_factor > 1:
                dims_new = np.round(
                    [dims_orig[0] / ds_factor, dims_orig[1] / ds_factor, dims_orig[2] / ds_factor]).astype(int)
                data_zyx = resize(data_zyx_raw, dims_new, order=1, preserve_range=True).astype(np.uint16)
                anisotropy = anisotropy_raw  # * dims_new[1] / dims_orig[1]
            else:
                data_zyx = data_zyx_raw.copy()
                anisotropy = anisotropy_raw

            if ("log" in model_name) or ("bkg" in model_name):
                im_log, im_bkg = calculate_LoG(data_zyx=data_zyx, scale_vec=pixel_res_raw)
            if "log" in model_name:
                data_zyx = im_log
            elif "bkg" in model_name:
                print("bkg")
                data_zyx = im_bkg

            assert ds_factor >= 1.0

            # Select 2D/3D behavior and set some parameters
            do_3D = data_zyx.shape[0] > 1

            # Preliminary checks on Cellpose model
            if pretrained_model is None:
                if model_type not in ["nuclei", "cyto2", "cyto"]:
                    raise ValueError(f"ERROR model_type={model_type} is not allowed.")
            else:
                if not os.path.exists(pretrained_model):
                    raise ValueError(f"{pretrained_model=} does not exist.")

            logging.info(
                f"mask will have shape {data_zyx.shape} "
            )

            # Initialize cellpose
            gpu = use_gpu()
            if pretrained_model:
                model = models.CellposeModel(
                    gpu=gpu, pretrained_model=pretrained_model
                )
            else:
                model = models.CellposeModel(gpu=gpu, model_type=model_type)

            if iter_i == 0:
                # Save metadata to a JSON file
                metadata_out_path = os.path.join(save_directory, "metadata.json")
                with open(metadata_out_path, 'w') as json_file:
                    json.dump(metadata, json_file)

            metadata["cellpose_model"] = model
            # Initialize other things
            logging.info(f"Start cellpose_segmentation task for {image_path}")
            logging.info(f"do_3D: {do_3D}")
            logging.info(f"use_gpu: {gpu}")
            logging.info(f"model_type: {model_type}")
            logging.info(f"pretrained_model: {pretrained_model}")
            logging.info(f"anisotropy: {anisotropy}")

            # Execute illumination correction
            image_mask, image_probs = segment_FOV(
                data_zyx,  # data_zyx.compute(),
                model=model,
                do_3D=do_3D,
                anisotropy=anisotropy,
                label_dtype=np.uint32,
                diameter=cell_diameter / ds_factor,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                min_size=min_size,
                pretrain_flag=(pretrained_model != None),
                return_probs=return_probs
            )

            if False: #ds_factor > 1.0:
                image_mask_out = resize(image_mask, dims_orig, order=0, anti_aliasing=False,
                                        preserve_range=True)
                image_probs_out = resize(image_probs, dims_orig, order=1, preserve_range=True)

            else:
                image_mask_out = image_mask
                image_probs_out = image_probs

            mask_zarr[t] = image_mask_out
            prob_zarr[t] = image_probs_out
            grad_zarr[t] = image_grads_out

            # im_name = image_path.replace('.zarr', '')
            # with TiffWriter(im_name + 'tif', bigtiff=True) as tif:
            #     tif.write(data_zyx)

            logging.info(f"End file save process, exit")

            iter_i += 1
        else:
            print("skipping " + label_path)

    return {}


if __name__ == "__main__":
    # sert some hyperparameters
    overwrite = False
    model_type = "cyto"
    output_label_name = "td-Tomato"
    seg_channel_label = "561"
    ds_factor = 2

    # set path to CellPose model to use
    pretrained_model = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\240219_LCP1_67hpf_to_\\cellpose\\models\\LCP-Multiset-v1"

    # set read/write paths
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    # project_name = "240219_LCP1_93hpf_to_127hpf"
    project_name = "230425_EXP21_LCP1_D6_1pm_DextranStabWound"

    cellpose_segmentation(root=root, project_name=project_name, return_probs=True, ds_factor=ds_factor,
                          pretrained_model=pretrained_model, overwrite=overwrite, adjust_background=True)
