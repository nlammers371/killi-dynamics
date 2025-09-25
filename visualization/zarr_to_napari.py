import zarr
import napari
from pathlib import Path
import numpy as np

# zarr_path = Path("E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/killi_tracker/built_data/zarr_image_files/20241126_LCP1-NLSMSC.zarr")
date = "20250716"
zarr_path = Path(f"/media/nick/cluster/projects/data/killi_tracker/built_data/zarr_image_files/{date}/")
image_list = sorted(list(zarr_path.glob("*_z.zarr")))

image_ind = 2

im_zarr = zarr.open(image_list[image_ind], mode='r')

time_range = [0, 5]
# scale_vec = tuple(im_zarr.attrs['voxel_size_um'])  # (z, y, x) spacing in microns
im_plot = np.squeeze(im_zarr[:, :])   # Assuming channel 0 is the nuclear channel
# im_LoG, thresh = calculate_li_thresh(im_plot, thresh_li=1, use_subsample=False)

viewer = napari.Viewer()
# viewer.add_image(im_LoG, scale=scale_vec)
viewer.add_image(im_plot, channel_axis=0, colormap=["magenta", "Green"])#, scale=scale_vec)
# viewer.add_labels(im_LoG > 750, scale=scale_vec)

if __name__ == '__main__':
    napari.run()
    print("Check")