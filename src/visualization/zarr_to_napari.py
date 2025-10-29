import zarr
import napari
import numpy as np
from pathlib import Path
from src.registration.virtual_fusion import VirtualFuseArray

zarr_path = Path(r"Y:/killi_dynamics/built_data/zarr_image_files/20251019_BC1-NLS_52-80hpf.zarr")

vf = VirtualFuseArray(
    zarr_path,
    use_gpu=True,      # make sure this is True
    interp="nearest",
    eval_mode=False,
)

scale_vec = tuple(vf.attrs['voxel_size_um'])  # (z, y, x) spacing in microns
im_plot = np.squeeze(vf[1598])  # Assuming channel 0 is the nuclear channel
# warm up
# vf[0]
viewer = napari.Viewer()
viewer.add_image(im_plot, channel_axis=0, colormap=["turbo", "gray"], scale=scale_vec)

if __name__ == '__main__':
    napari.run()
    print("Check")