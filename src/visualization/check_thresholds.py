import napari
from pathlib import Path
import numpy as np
import skimage as ski
import SimpleITK as sitk
from skimage.measure import label
from skimage import morphology, segmentation
from scipy import ndimage as ndi

from src.data_io.zarr_utils import open_experiment_array

project_name = "MEM_NLS_test"
data_root = r"E:\pipeline_dev\killi_dynamics"
im, _store_path, _resolved_side = open_experiment_array(Path(data_root), project_name)
ch = 1
plot_frame = 30
scale_vec = tuple(im.attrs['voxel_size_um'])  # (z, y, x) spacing in microns
im_plot = np.squeeze(im[plot_frame, ch])   # Assuming channel 0 is the nuclear channel
# mask_plot = np.squeeze(mask_zarr[22])
print("Calculating LoG...")

viewer = napari.Viewer()
# viewer.add_image(im_LoG, scale=scale_vec)
viewer.add_image(im_plot,  scale=scale_vec)
# viewer.add_image(data_log_i, scale=scale_vec)
# viewer.add_labels(markers, scale=scale_vec)


if __name__ == '__main__':
    napari.run()
    print("Check")