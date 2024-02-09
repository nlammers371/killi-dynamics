import napari
import os
import skimage.io as io
import glob2 as glob

from czitools import misc_tools
# set parameters
image_folder ="E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\Nick\killi_tracker\\built_data\\231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw\\cellpose\\"
image_list = glob.glob(image_folder + "*.tif")
image_ind = 0

im = io.imread(image_list[image_ind])



if __name__ == '__main__':
    napari.run()