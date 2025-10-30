import napari
from pathlib import Path
import numpy as np
import skimage as ski
import SimpleITK as sitk
from skimage.measure import label
from skimage import morphology, segmentation
from scipy import ndimage as ndi
from src.data_io.zarr_utils import open_experiment_array
from src.segmentation.li_thresholding import compute_li_threshold_single_frame

data_root = r"Y:\killi_dynamics"
project_name = "20251019_BC1-NLS_52-80hpf"
im, _store_path, _resolved_side = open_experiment_array(Path(data_root), project_name)
ch = 1
plot_frame = 1000
thresh = 71
scale_vec = tuple(im.attrs['voxel_size_um'])  # (z, y, x) spacing in microns
im_plot = np.squeeze(im[plot_frame, ch])   # Assuming channel 0 is the nuclear channel
# mask_plot = np.squeeze(mask_zarr[22])
print("Calculating LoG...")
im_LoG, thresh = compute_li_threshold_single_frame(im_plot, thresh_li=thresh)


viewer = napari.Viewer()
viewer.add_image(im_LoG, scale=scale_vec)
viewer.add_image(im_plot,  scale=scale_vec)
# viewer.add_image(data_log_i, scale=scale_vec)
viewer.add_labels(label(im_LoG >= thresh), scale=scale_vec)


if __name__ == '__main__':
    napari.run()
    print("Check")