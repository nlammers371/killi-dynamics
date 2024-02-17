import napari
import os
import skimage.io as io
from skimage.transform import resize
import glob2 as glob
# from skimage import label
import numpy as np
import json
from src.utilities.functions import sphereFit, cart_to_sphere
from skimage.measure import label, regionprops

root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\"
project_name = "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
prob_thresh = -8

# ger list of images
image_folder = os.path.join(root, project_name, "")
image_list = sorted(glob.glob(os.path.join(image_folder + "*.tiff")))

# prob datasets
label_folder = os.path.join(root, "cellpose_output", project_name, "")
prob_list = sorted(glob.glob(os.path.join(label_folder + "*_probs.tif")))

# load metadata
metadata_file_path = os.path.join(label_folder, "metadata.json")
f = open(metadata_file_path)
metadata = json.load(f)

pixel_res_prob = np.asarray([metadata["PhysicalSizeZ"], metadata["PhysicalSizeY"], metadata["PhysicalSizeX"]])

metadata_out_path = os.path.join(label_folder, "metadata.json")
with open(metadata_out_path, 'w') as json_file:
    json.dump(metadata, json_file)


iter_i = 0

# Load prob image
raw_image = io.imread(image_list[iter_i])
prob_image = io.imread(prob_list[iter_i])
prob_mask = prob_image >= prob_thresh
label_image = label(prob_mask)

# generate reference grid
z_ref, y_ref, x_ref = np.meshgrid(range(prob_mask.shape[0]), range(prob_mask.shape[1]), range(prob_mask.shape[2]),
                                  indexing="ij")

nc_z = z_ref[prob_mask]
nc_y = y_ref[prob_mask]
nc_x = x_ref[prob_mask]

# fit to sphere
center_point = np.empty((3, ))
radius, center_point[2], center_point[1], center_point[0] = sphereFit(nc_x, nc_y, nc_z)

# now use the sphere fit to ID inlier and outlier points
# regions = regionprops(label_image) #, ["centroid", "coords", "area"])
# centroid_array = np.asarray([r["centroid"] for r in regions]) - center_point
# centroid_array_sph = cart_to_sphere(centroid_array[:, ::-1])

# convert reference arrays to spherical coordinages
n_px = x_ref.size
rpt_ref = cart_to_sphere(np.concatenate((x_ref.reshape(n_px, 1)-center_point[2],
                                         y_ref.reshape(n_px, 1)-center_point[1],
                                         z_ref.reshape(n_px, 1)-center_point[0]), axis=1))
r_ref = rpt_ref[:, 0]. reshape(x_ref.shape)
p_ref = rpt_ref[:, 1]. reshape(x_ref.shape)
t_ref = rpt_ref[:, 2]. reshape(x_ref.shape)

# subset to just those points that are part of a cell
r_ref_vec = r_ref[prob_mask]
p_ref_vec = p_ref[prob_mask]
t_ref_vec = t_ref[prob_mask]
label_vec = label_image[prob_mask]

# define bounds
grid_res = 250
r_grid = np.linspace(np.min(r_ref), np.max(r_ref), grid_res)
phi_grid = np.linspace(np.min(p_ref), np.max(p_ref), grid_res)
theta_grid = np.linspace(np.min(t_ref), np.max(t_ref), grid_res)

theta_res = np.abs(theta_grid[1]-theta_grid[0]) / 2
phi_res = np.abs(phi_grid[1]-phi_grid[0]) / 2

intersected_label_list = []
radial_dist_list = []
overlap_flag_vec = []
counter = 0
for p, phi in enumerate(phi_grid):
    for t, theta in enumerate(theta_grid):
        # flag masks with nearby centroids
        candidate_mask = (np.abs(t_ref_vec-theta) < theta_res) & (np.abs(p_ref_vec-phi) < phi_res)
        labels = label_vec[np.where(candidate_mask == 1)]

        label_i = np.unique(labels)
        intersected_label_list.append(label_i)

        r_list = []
        for lbi in label_i:
            r = np.mean(r_ref_vec[candidate_mask & (label_vec == lbi)])
            r_list.append(r)
        radial_dist_list.append(r_list)

        if len(label_i) > 1:
            overlap_flag_vec.append(True)
        else:
            overlap_flag_vec.append(False)

        counter += 1
        print(counter)

# iterate through and compile a list of all labels that were second
overlap_flag_vec = np.asarray(overlap_flag_vec)
shadow_label_list = []
for i in range(len(radial_dist_list)):
    lbs = intersected_label_list[i]
    rs = radial_dist_list[i]
    if len(lbs) > 1:
        si = np.argsort(rs)
        for s in si[1:]:
            shadow_label_list.append(lbs[s])


shadow_label_vec = np.unique(np.asarray(shadow_label_list))

prob_mask_u = prob_mask.copy().astype(int)
for sh in shadow_label_vec:
    prob_mask_u[label_image == sh] = 2
# convert centroids and mask coordinates to spherical coor
# r_image = np.sqrt((z_ref-center_point[0])**2 + (y_ref-center_point[1])**2 + (x_ref-center_point[2])**2)


viewer = napari.view_image(raw_image)
viewer.add_labels(prob_mask_u)
# viewer.add_image(r_image, colormap="magma", opacity=0.4)


if __name__ == '__main__':
    napari.run()