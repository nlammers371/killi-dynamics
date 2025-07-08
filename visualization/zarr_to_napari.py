import napari
import numpy as np
import zarr

zarr_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\killi_tracker\\built_data\\zarr_image_files\\20250419_BC1-NLSMSC_side1.zarr"
image_data = zarr.open(zarr_path, mode="r")
scale_vec = tuple([image_data.attrs['PhysicalSizeZ'],
                  image_data.attrs['PhysicalSizeY'],
                  image_data.attrs['PhysicalSizeX']])

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(np.squeeze(image_data[100:105]), channel_axis=0, scale=scale_vec)

if __name__ == '__main__':
    napari.run()