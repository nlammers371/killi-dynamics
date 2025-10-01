import zarr
import napari
from pathlib import Path
import numpy as np
from src.nucleus_dynamics.utilities.image_utils import calculate_LoG

# zarr_path = Path("E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/killi_tracker/built_data/zarr_image_files/20241126_LCP1-NLSMSC.zarr")
date = "20250716"
zarr_path = Path(f"/media/nick/cluster/projects/data/killi_tracker/built_data/zarr_image_files/{date}/")
image_list = sorted(list(zarr_path.glob("*.zarr")))

# zarr_path = "/media/nick/cluster/projects/data/killi_tracker/built_data/cellpose_output/tdTom-bright-log-v5/20250716/20250716_well0000_probs.zarr"
# image_ind = 0
# im_zarr = zarr.open(image_list[image_ind], mode='r')
im_zarr = zarr.open(zarr_path, mode='r')
scale_vec = tuple(im_zarr.attrs['voxel_size_um'])  # (z, y, x) spacing in microns
im_plot = np.squeeze(im_zarr[12])   # Assuming channel 0 is the nuclear channel

# im_log, im_bkg = calculate_LoG(im_plot, scale_vec=scale_vec, sigma_dims=[1, 3, 3], subtract_bkg=False, log_sigma=1)
# im_LoG, thresh = calculate_li_thresh(im_plot, thresh_li=1, use_subsample=False)

viewer = napari.Viewer()
# viewer.add_image(im_LoG, scale=scale_vec)
viewer.add_image(im_plot, scale=scale_vec)
# viewer.add_image(im_log, scale=scale_vec, contrast_limits=(0, 65000))
# viewer.add_image(im_bkg, scale=scale_vec, contrast_limits=(0, 65000))
# viewer.add_labels(im_LoG > 750, scale=scale_vec)

if __name__ == '__main__':
    napari.run()
    print("Check")