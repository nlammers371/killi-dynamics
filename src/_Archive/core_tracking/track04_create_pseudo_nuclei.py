# import napari
import os
import skimage.io as io
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from ultrack.imgproc.intensity import robust_invert
from ultrack.utils import estimate_parameters_from_labels, labels_to_edges
from ultrack.imgproc.segmentation import detect_foreground
from skimage.measure import regionprops
from skimage.morphology import ball, dilation
import zarr
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
import json
import glob2 as glob

def create_pseudo_nuclei(root, project_name, overwrite_flag=True,  out_shape=None, center_rad=3):

    # get path to full masks
    mask_zarr_path = os.path.join(root, "built_data", "cleaned_cell_labels", project_name + ".zarr")
    mask_zarr = zarr.open(mask_zarr_path, mode='r')
    if out_shape is None:
        out_shape = mask_zarr.shape
    # create new zarr to store nuclei
    out_zarr_path = os.path.join(root, "built_data", "cleaned_cell_labels",  project_name + "_centroids.zarr")
    dtype = mask_zarr.dtype

    if (not os.path.exists(out_zarr_path)) | overwrite_flag:
        out_zarr = zarr.open(out_zarr_path, mode='w', shape=out_shape, dtype=dtype,
                             chunks=(1,) + mask_zarr.shape[1:])
    else:
        out_zarr = zarr.open(out_zarr_path, mode='a', chunks=(1,) + mask_zarr.shape[1:])

    fp = ball(center_rad)
    # Load and resize
    print("Loading time points...")
    for m in tqdm(range(mask_zarr.shape[0])):

        lb_flag = np.any(out_zarr[m] > 0)

        if (not lb_flag) | overwrite_flag:
            frame_arr = np.asarray(mask_zarr[m])
            regions = regionprops(frame_arr)
            new_mask_array = np.zeros(frame_arr.shape, dtype=dtype)
            for region in regions:
                centroid = np.asarray(region.centroid).astype(int)
                new_mask_array[centroid[0], centroid[1], centroid[2]] = region.label

            new_mask = dilation(new_mask_array, fp)
            out_zarr[m] = new_mask

    return {}


if __name__ == '__main__':

    # set path to mask files
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "240219_LCP1_93hpf_to_127hpf" #"231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"  #

    create_pseudo_nuclei(root, project_name, overwrite_flag=False, out_shape=(1350, 170, 412, 412))


