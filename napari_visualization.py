import napari
import os
from aicsimageio import AICSImage
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from ultrack.imgproc.intensity import robust_invert
from ultrack.imgproc.segmentation import detect_foreground
import time
from ultrack.utils.array import array_apply
from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr

from czitools import misc_tools
# set parameters
root = "D:\\Syd\\231016_EXP40_LCP1_UVB_300mJ\\PreUVB_Timelapse_Raw\\"
filename_root = "e2_LCP1_preZap_Timelapse_2023_10_16__20_29_18_539"
ds_factor = 4
config_path = "/_Archive/kf_config.txt"
# specify time points to load
time_points = np.arange(1, 51)
image_list = []
# Load and resize
print("Loading time points...")
for t, time_point in enumerate(tqdm(time_points)):
    readPath = os.path.join(root, filename_root + f"({time_point}).czi")

    imObject = AICSImage(readPath)
    image_data = np.squeeze(imObject.data)
    dims_orig = image_data.shape
    dims_new = np.round([dims_orig[0] / ds_factor * 2, dims_orig[1] / ds_factor, dims_orig[2] / ds_factor]).astype(int)
    data_zyx = resize(image_data, dims_new, order=1)
    if t == 0:
        scale_vec = np.asarray(imObject.physical_pixel_sizes)
        scale_vec = scale_vec*ds_factor
        scale_vec[0] = scale_vec[0]/2

        data_tzyx = np.empty((len(time_points), data_zyx.shape[0], data_zyx.shape[1], data_zyx.shape[2]), dtype=data_zyx.dtype)

    data_tzyx[t, :, :, :] = data_zyx


# segment
start = time.time()
detection = np.empty(data_tzyx.shape, dtype=np.uint)
array_apply(
    data_tzyx,
    out_array=detection,
    func=detect_foreground,
    sigma=15.0,
    voxel_size=scale_vec,
)

boundaries = np.empty(data_tzyx.shape, dtype=np.uint)
array_apply(
    data_tzyx,
    out_array=boundaries,
    func=robust_invert,
    voxel_size=scale_vec,
)


print("Examine segmentation in napari")
# viewer = napari.view_image(data_tzyx, scale=tuple(scale_vec))
# label_layer = viewer.add_labels(detection, name='segmentation', scale=tuple(scale_vec))
# boundary_layer = viewer.add_image(boundaries, visible=False, scale=tuple(scale_vec))
# viewer.theme = "dark"

cfg = load_config(config_path)
# cfg =  MainConfig()  # or load default config
cfg.segmentation_config.threshold = 0.5
print(cfg)

# Perform tracking
track(
    cfg,
    detection=detection,
    edges=boundaries,
    scale=scale_vec,
    overwrite=True,
)

print("saving downsampled image data...")
np.save(project_path + "image_data_ds.npy", data_tzyx)

if __name__ == '__main__':
    napari.run()