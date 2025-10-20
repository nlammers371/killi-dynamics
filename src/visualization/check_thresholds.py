import zarr
import napari
from pathlib import Path
import numpy as np
import skimage as ski
import SimpleITK as sitk
from skimage.measure import label
from skimage import morphology, segmentation
from scipy import ndimage as ndi

project_name = "MEM_NLS_test"
data_root = r"E:\pipeline_dev\killi_dynamics"
zarr_path = Path(data_root) / "built_data" / "zarr_image_files"/ f"{project_name}.zarr"
im = zarr.open(zarr_path, mode='r')
ch = 1
plot_frame = 30
scale_vec = tuple(im.attrs['pixel_size_um'])  # (z, y, x) spacing in microns
im_plot = np.squeeze(im[plot_frame, ch])   # Assuming channel 0 is the nuclear channel
# mask_plot = np.squeeze(mask_zarr[22])
print("Calculating LoG...")
# gauss_sigma = (1.33, 4, 4)
# gaussian_background = ski.filters.gaussian(im_plot, sigma=gauss_sigma, preserve_range=True)
# data_bkg = im_plot - gaussian_background
#
# data_log = sitk.GetArrayFromImage(
#     sitk.LaplacianRecursiveGaussian(sitk.GetImageFromArray(data_bkg), sigma=1)
# )
# data_log_i = ski.util.invert(data_log)
#
# print("Calculating markers...")
# h = 0.02 * data_log_i.max()  # adjust scale
# markers = morphology.h_maxima(data_log_i, h)
# markers = ndi.label(markers)[0]

viewer = napari.Viewer()
# viewer.add_image(im_LoG, scale=scale_vec)
viewer.add_image(im_plot,  scale=scale_vec)
# viewer.add_image(data_log_i, scale=scale_vec)
# viewer.add_labels(markers, scale=scale_vec)


if __name__ == '__main__':
    napari.run()
    print("Check")