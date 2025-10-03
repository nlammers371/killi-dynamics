import numpy as np
from skimage.transform import resize
import skimage as ski
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

def fill_zeros_nearest(arr: np.ndarray) -> np.ndarray:
    """
    Replace all zero pixels in `arr` with the value of the nearest nonzero pixel.
    Works for nD arrays.
    """
    mask = arr == 0
    if not np.any(mask):
        return arr

    # distance_transform_edt returns the index of the nearest nonzero for each zero voxel
    # return_indices=True gives us a tuple of index arrays (one per axis)
    _, indices = distance_transform_edt(mask, return_distances=True, return_indices=True)
    filled = arr[tuple(indices)]
    return filled


# function to process image stacks to account for uneven intensities and get lap-of-gaussian
def calculate_LoG(data_zyx, scale_vec, make_isotropic=False, subtract_background=False, sigma_dims=None):

    if sigma_dims is None:
        sigma_dims = [1.5, 4.5, 4.5]

    # estimate background using blur
    top1 = np.percentile(data_zyx, 99.9)
    data_capped = data_zyx.copy()
    data_capped[data_capped > top1] = top1
    data_capped = fill_zeros_nearest(data_capped) # fill in zeros with nearest non-zero
    # data_capped = data_capped[:, 500:775, 130:475]

    shape_orig = np.asarray(data_capped.shape)
    shape_iso = shape_orig.copy()
    iso_factor = scale_vec[0] / scale_vec[1]
    shape_iso[0] = shape_iso[0] * iso_factor

    gaussian_background = ski.filters.gaussian(data_capped, sigma=(sigma_dims[0], sigma_dims[1], sigma_dims[2]),
                                               preserve_range=True, truncate=4,
                                               mode="reflect")
    if not subtract_background:
        data_1 = np.divide(data_capped, gaussian_background)
    else:
        data_1 = data_capped - gaussian_background
    data_rs = resize(data_1, shape_iso, preserve_range=True, order=1)
    image = sitk.GetImageFromArray(data_rs)
    data_log = sitk.GetArrayFromImage(sitk.LaplacianRecursiveGaussian(image, sigma=1))
    if not make_isotropic:
        data_log_i = resize(ski.util.invert(data_log), shape_orig, preserve_range=True, order=1)
        # data_log_i = ski.util.invert(data_log_i)
    else:
        data_log_i = ski.util.invert(data_log)

    # data_log_i[data_rs==0] = np.min(data_log_i)
    # rescale and convert to 16 bit
    if make_isotropic:
        data_bkg_16 = data_rs.copy()
    else:
        data_bkg_16 = data_1.copy()
    data_bkg_16 = data_bkg_16 - np.min(data_bkg_16)
    data_bkg_16 = np.round(data_bkg_16 / np.max(data_bkg_16) * 2 ** 16 - 1).astype(np.uint16)

    log_i_16 = data_log_i.copy()
    log_i_16 = log_i_16 - np.min(log_i_16)
    log_i_16 = np.round(log_i_16 / np.max(log_i_16) * 2 ** 16 - 1).astype(np.uint16)
    if not make_isotropic:
        log_i_16[data_capped == 0] = 0
    else:
        log_i_16[data_rs == 0] = 0

    return log_i_16, data_bkg_16