import napari
import numpy as np
import zarr
from pathlib import Path


zarr_root = Path("Y:\\projects\\data\killi_tracker\\built_data\\zarr_image_files\\20250716")
well_ind = 3
zarr_path = zarr_root / f"20250716_well{well_ind:04d}_z.zarr"
image_data = zarr.open(zarr_path, mode="r")
# scale_vec = tuple([image_data.attrs['PhysicalSizeZ'],
#                   image_data.attrs['PhysicalSizeY'],
#                   image_data.attrs['PhysicalSizeX']])

viewer = napari.Viewer(ndisplay=2)
viewer.add_image(image_data, channel_axis=0)

if __name__ == '__main__':
    napari.run()