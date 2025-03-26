import zarr
import napari
import os
import numpy as np
from src.build_killi.run02_segment_nuclei import calculate_li_thresh
import pandas as pd
import statsmodels.api as sm
import os
from skimage.measure import label, regionprops
from scipy.interpolate import interp1d

os.environ["QT_API"] = "pyqt5"

# get filepaths
root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
project = "20250311_LCP1-NLSMSC_side2"
# project2 = "20250311_LCP1-NLSMSC_side2"
zpath = os.path.join(root, "built_data", "zarr_image_files", project + ".zarr")
mpath = os.path.join(root, "built_data", "mask_stacks", project + "_mask_stacks.zarr")
# zpath2 = os.path.join(root, "built_data", "zarr_image_files", project2 + ".zarr")
# load
im_full = zarr.open(zpath, mode="r")
mask_full = zarr.open(mpath, mode="r")

# generate frame indices
t_int = 2300
nucleus_channel = 1

# get scale info
scale_vec = tuple([im_full.attrs['PhysicalSizeZ'], im_full.attrs['PhysicalSizeY'], im_full.attrs['PhysicalSizeX']])

print("Loading zarr files...")
# extract relevant frames
im = np.squeeze(im_full[t_int, nucleus_channel])
mask = np.squeeze(mask_full[t_int])
# li_df = pd.read_csv(os.path.join(root, "built_data", "mask_stacks", project + "_li_df.csv"))
# li_df_raw1 = pd.read_csv(os.path.join(root, "built_data", "mask_stacks", project + "_li_df_raw.csv"))
# li_df_raw2 = pd.read_csv(os.path.join(root, "built_data", "mask_stacks", "20250311_LCP1-NLSMSC_side2" + "_li_df_raw.csv"))
#
# li_df_raw = pd.concat([li_df_raw1, li_df_raw2], axis=0, ignore_index=True)
#
# # Example data: x is the time variable, y is your timeseries with outliers
# x = li_df_raw["frame"].to_numpy()
# y = li_df_raw["li_thresh"].to_numpy()
#
# y_thresh = np.percentile(y, 95) / 10
#
# outlier_filter = y > y_thresh
# x = x[outlier_filter]
# y = y[outlier_filter]
# si = np.argsort(x)
# x = x[si]
# y = y[si]
# # Fit quantile regression at the 95th percentile
# # coefficients = np.polyfit(x, y, deg=3)
# # poly_model = np.poly1d(coefficients)
# lowess_result = sm.nonparametric.lowess(y, x, frac=0.3, it=3)
# x_lowess = lowess_result[:, 0]
# y_lowess = lowess_result[:, 1]
#
# frames_full = li_df["frame"].to_numpy()
# frames_to_fit = frames_full[(frames_full>=np.min(x)) & (frames_full<=np.max(x))]
# # thresh_predictions = poly_model(frames_to_fit)
# thresh_interp = interp1d(x_lowess, y_lowess, kind="linear", fill_value="extrapolate")
# thresh_predictions = thresh_interp(frames_full)
# # generate mask
# li_thresh = thresh_predictions[t_int]
# im_log, _ = calculate_li_thresh(im, thresh_li=li_thresh)
# mask = label(im_log >= li_thresh)
#
# # look for highly eccentric masks
# props = regionprops(mask, spacing=scale_vec)
# eig_array = np.sqrt(np.asarray([p["inertia_tensor_eigvals"] for p in props]))
# # eig_ratio = np.divide(eig_array[:, 2], eig_array[:, 0])
#
# eig_flags = eig_array[:, 2] < 2
# eig_labels = np.asarray([props[p].label for p in range(len(props)) if eig_flags[p]])
# spindrel_mask = np.isin(mask, eig_labels)

viewer = napari.Viewer()

viewer.add_image(im, scale=scale_vec,  colormap="gray", contrast_limits=[0, 2500])
viewer.add_labels(mask, scale=scale_vec)
# viewer.add_labels(spindrel_mask, scale=scale_vec)
# viewer.add_labels(mask01, scale=scale_vec)
# viewer.add_labels(shadow_mask, scale=scale_vec)
napari.run()

