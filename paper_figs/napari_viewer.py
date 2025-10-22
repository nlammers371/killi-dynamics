import zarr
import napari
from pathlib import Path
from src.build_killi.run02_segment_nuclei import calculate_li_thresh
import numpy as np

# I used this to manually mark the center of the wound in XY
zarr_path = Path("E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/killi_tracker/built_data/zarr_image_files/20241126_LCP1-NLSMSC_mip.zarr")
im_zarr = zarr.open(zarr_path, mode='r')

# time_range = [500, 505]
scale_vec = tuple([im_zarr.attrs['PhysicalSizeZ'], im_zarr.attrs['PhysicalSizeY'], im_zarr.attrs['PhysicalSizeX']])

# thresh = 185 #200
# # im_plot = np.squeeze(im_zarr[time_range[0]:time_range[-1]+1, 1])
# im_plot = np.squeeze(im_zarr[time_range[0]:time_range[-1]+1])   # Assuming channel 0 is the nuclear channel
# im_LoG, thresh = calculate_li_thresh(im_plot, thresh_li=1, use_subsample=False)

viewer = napari.Viewer()
# viewer.add_image(im_LoG, scale=scale_vec)
viewer.add_image(im_zarr[:200], channel_axis=1, scale=scale_vec)
# viewer.add_labels(im_LoG > 750, scale=scale_vec)

if __name__ == '__main__':
    napari.run()
    print("Check")