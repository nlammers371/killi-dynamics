import napari
import os
import skimage.io as io
from skimage.transform import resize
import glob2 as glob
import skimage
from skimage import data
from skimage import color, morphology
from skimage import exposure, util
import numpy as np
from skimage import (
    data, restoration, util
)
# from czitools import misc_tools
# set parameters
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\"
# root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/killi_tracker/built_data/"
project_name = "240219_LCP1_67hpf_to_" #"231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
image_folder = os.path.join(root, project_name, "")
image_list = sorted(glob.glob(os.path.join(image_folder + "*.tiff")))
label_folder = os.path.join(root, "cellpose_output", project_name, "")

# label_list = sorted(glob.glob(os.path.join(label_folder + "*_labels.tif")))
prob_list = sorted(glob.glob(os.path.join(label_folder + "*_probs.tif")))
image_ind = 0

im = io.imread(image_list[image_ind])
prob = io.imread(prob_list[image_ind])


# print("performing background correction...")

# background = restoration.rolling_ball(
#     im.astype(np.float64),
#     kernel=restoration.ellipsoid_kernel(
#         kernel_size,
#         0.1
#     )
# )

# background = skimage.filters.gaussian(im.astype(np.float64), sigma=10)

# im_rs = resize(im, (im.shape[0]//2, im.shape[1]//2, im.shape[2]//2), preserve_range=True).astype(np.uint16)
# res = morphology.white_tophat(im_rs, footprint)
# im_c = np.divide(im.astype(np.float64), background.astype(np.float64))
# im_c2 = im.astype(np.float64) - background.astype(np.float64)

# print("Done.")
# lb = io.imread(label_list[image_ind])
# pr = io.imread(prob_list[image_ind])
# im = im / np.max(im)
#

viewer = napari.view_image(im, scale=tuple([1, 1, 1]))
viewer.add_labels(prob > -8, name="thresh", scale=tuple([2, 2, 2]))

# labels = viewer.add_labels(lb, name="labels")


if __name__ == '__main__':
    napari.run()