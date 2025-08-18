import napari
import numpy as np
import zarr
from pathlib import Path


# zarr_path = Path("I:\\Nick\\killi_tracker\\built_data\\zarr_image_files\\20250419_BC1-NLSMSC_fused.zarr")
# zarr_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\mask_stacks\\20250419_BC1-NLSMSC_mask_fused.zarr"
zarr_path = "I:\\Nick\\killi_tracker\\built_data\\zarr_image_files\\20250419_BC1-NLSMSC_fused.zarr"
# well_ind = 0
# zarr_path = zarr_root / f"20251126_LCP1-NLSMSC.zarr"
image_data = zarr.open(zarr_path, mode="r")
# scale_vec = tuple([image_data.attrs['PhysicalSizeZ'],
#                   image_data.attrs['PhysicalSizeY'],
#                   image_data.attrs['PhysicalSizeX']])

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(image_data[-4:], channel_axis=1, scale=(3, 0.85, 0.85))
# viewer.add_labels(image_data[485:490], scale=(3, 0.85, 0.85))

if __name__ == '__main__':
    napari.run()
    print("Pause")