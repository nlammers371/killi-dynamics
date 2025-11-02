import numpy as np
from pathlib import Path
from skimage.measure import label
from scipy.ndimage import rotate
import napari

from src.registration.virtual_fusion import VirtualFuseArray

def rotate_half_stack(im_mask: np.ndarray, Zm: int, rot_angle_deg: float, order: int=0, which_half: str = "second") -> np.ndarray:
    """
    Rotate one half of a 3D (Z, Y, X) mask around the Z axis by a given angle.

    Parameters
    ----------
    im_mask : np.ndarray
        Input 3D array (Z, Y, X).
    rot_angle_deg : float
        Rotation angle in degrees. Positive = counter-clockwise in XY.
    which_half : {"first", "second"}
        Which half of the Z stack to rotate.
        "first"  → rotate im_mask[:Zm, :, :]
        "second" → rotate im_mask[Zm:, :, :]

    Returns
    -------
    np.ndarray
        Rotated and recombined 3D mask, same shape as input.
    """
    if im_mask.ndim != 3:
        raise ValueError("im_mask must be 3D (Z, Y, X)")

    # Zm = im_mask.shape[0] // 2
    first_half = im_mask[:Zm, :, :]
    second_half = im_mask[Zm:, :, :]

    if rot_angle_deg == 0:
        return im_mask.copy()

    if which_half == "first":
        first_half = rotate(first_half, angle=rot_angle_deg, axes=(1, 2),
                            reshape=False, order=order, mode="nearest")
    elif which_half == "second":
        second_half = rotate(second_half, angle=rot_angle_deg, axes=(1, 2),
                             reshape=False, order=order, mode="nearest")
    else:
        raise ValueError("which_half must be 'first' or 'second'")

    # Recombine
    return np.concatenate([first_half, second_half], axis=0)

if __name__ == "__main__":

    # --- USER CONFIG ---
    rot_angle_deg = -5.0        # degrees to rotate im_mask around Z axis
    rotate_mask = True           # toggle rotation
    interp_order = 0             # 0 = nearest (fastest, keeps labels)
    # -------------------

    # load the fused array
    zarr_path = Path(r"Y:/killi_dynamics/built_data/zarr_image_files/20251019_BC1-NLS_52-80hpf.zarr")
    vf = VirtualFuseArray(
        zarr_path,
        use_gpu=True,
        interp="nearest",
        eval_mode=False,
    )

    # extract voxel scale and target plane
    scale_vec = tuple(vf.attrs["voxel_size_um"])  # (z, y, x)
    im_plot = np.squeeze(vf[1530, 0])             # single timepoint, channel 1

    # threshold and label
    # im_mask = label(im_plot > 2500)

    # assume im_mask is your 3D label array (Z, Y, X)
    Zm = 188 #im_mask.shape[0] // 2
    # print("Rotating second half by +12 degrees...")
    im_rot = rotate_half_stack(im_plot, Zm=Zm, rot_angle_deg=rot_angle_deg, order=1, which_half="first")
    # # optionally rotate mask around z-axis
    # print(f"Rotating mask by {rot_angle_deg}° around Z axis...")
    # im_mask = rotate(
    #     im_mask,
    #     angle=rot_angle_deg,
    #     axes=(1, 2),         # rotate in (Y,X) plane
    #     reshape=False,       # keep same shape
    #     order=0,   # 0 = nearest for labels
    #     mode="nearest"
    # )
    #
    # visualize
    viewer = napari.Viewer()
    viewer.add_image(im_plot, scale=scale_vec)
    # viewer.add_labels(im_mask, scale=scale_vec)
    viewer.add_image(im_rot, scale=scale_vec)
    napari.run()
