import napari
import os
import skimage.io as io
from skimage.transform import resize
import glob2 as glob
import numpy as np
# from czitools import misc_tools
# set parameters
image_folder = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw\\"
image_list = sorted(glob.glob(os.path.join(image_folder + "*.tiff")))
label_folder = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\cellpose_output_test2\\231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw\\"

label_list = sorted(glob.glob(os.path.join(label_folder + "*_labels.tif")))
prob_list = sorted(glob.glob(os.path.join(label_folder + "*_probs.tif")))
image_ind = 0

im = io.imread(image_list[image_ind])
lb = io.imread(label_list[image_ind])
pr = io.imread(prob_list[image_ind])
# im = im / np.max(im)
# im_rs = resize(im, (im.shape[0]*4, im.shape[1]*4), preserve_range=True).astype(np.uint16)
viewer = napari.view_image(im)
probs = viewer.add_image(pr, name="probabilities", contrast_limits=[-8, 8])
labels = viewer.add_labels(lb, name="labels")


if __name__ == '__main__':
    napari.run()