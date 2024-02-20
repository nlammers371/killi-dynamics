import napari
import os
import skimage.io as io
from skimage.transform import resize
import glob2 as glob
# from skimage import label
import skimage
from skimage.morphology import ball
import numpy as np
import json
from src.utilities.functions import sphereFit, cart_to_sphere
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt


root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\"
project_name = "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
prob_thresh = -8

# ger list of images
image_folder = os.path.join(root, project_name, "")
image_list = sorted(glob.glob(os.path.join(image_folder + "*.tiff")))

# prob datasets
label_folder = os.path.join(root, "cellpose_output", project_name, "")
prob_list = sorted(glob.glob(os.path.join(label_folder + "*_probs.tif")))

print("loading data and making image masks...")
# load metadata
metadata_file_path = os.path.join(label_folder, "metadata.json")
f = open(metadata_file_path)
metadata = json.load(f)

pixel_res_prob = np.asarray([metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])

metadata_out_path = os.path.join(label_folder, "metadata.json")
with open(metadata_out_path, 'w') as json_file:
    json.dump(metadata, json_file)


iter_i = 0

# Load prob image
raw_image = io.imread(image_list[iter_i])
prob_image = io.imread(prob_list[iter_i])
if project_name == "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw":
    resize(prob_image, (prob_image.shape[0] // 2, prob_image.shape[1] // 2, prob_image.shape[2] // 2), preserve_range=True)

raw_image = resize(raw_image, prob_image.shape, preserve_range=True)
prob_mask = prob_image >= prob_thresh

# erode
ball_rad = np.ceil(3 / pixel_res_prob[0])
fp = ball(ball_rad)
prob_mask_er = skimage.morphology.binary_erosion(prob_mask, fp)

# label_image_out = label(prob_mask)
label_image = label(prob_mask_er)  # use this version to filter

# generate reference grid
print("calculating best-fit sphere...")
z_ref, y_ref, x_ref = np.meshgrid(range(prob_mask.shape[0]), range(prob_mask.shape[1]), range(prob_mask.shape[2]),
                                  indexing="ij")
z_ref = z_ref*pixel_res_prob[0]
y_ref = y_ref*pixel_res_prob[1]
x_ref = x_ref*pixel_res_prob[2]

# get location of masked pixels
nc_z = z_ref[prob_mask]
nc_y = y_ref[prob_mask]
nc_x = x_ref[prob_mask]

# fit to sphere
center_point = np.empty((3, ))
radius, center_point[2], center_point[1], center_point[0] = sphereFit(nc_x, nc_y, nc_z)

# now use the sphere fit to ID inlier and outlier points
# centroid_array = np.asarray([r["centroid"] for r in regions]) - center_point
# centroid_array_sph = cart_to_sphere(centroid_array[:, ::-1])

# remove small regions
regions = regionprops(label_image) #, ["centroid", "coords", "area"])
area_vec = np.asarray([r["area"] for r in regions])*np.prod(pixel_res_prob)
rm_indices = np.where(area_vec < 270)[0]
for ind in rm_indices:
    prob_mask_er[label_image == ind] = 0
    label_image[label_image == ind] = 0

# convert reference arrays to spherical coordinages
n_px = x_ref.size
rpt_ref = cart_to_sphere(np.concatenate((x_ref.reshape(n_px, 1)-center_point[2],
                                         y_ref.reshape(n_px, 1)-center_point[1],
                                         z_ref.reshape(n_px, 1)-center_point[0]), axis=1))
r_ref = rpt_ref[:, 0]. reshape(x_ref.shape)
p_ref = rpt_ref[:, 1]. reshape(x_ref.shape)
t_ref = rpt_ref[:, 2]. reshape(x_ref.shape)

print("casting rays to check for refraction artifacts...")
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

theta_res = np.abs(theta_grid[1]-theta_grid[0]) / 3
phi_res = np.abs(phi_grid[1]-phi_grid[0]) / 3

intersected_label_list = []
radial_dist_list = []
overlap_flag_vec = []
counter = 0
pass_i = 0
for p, phi in enumerate(phi_grid):
    for t, theta in enumerate(theta_grid):
        # flag masks with nearby centroids
        candidate_mask = (np.abs(t_ref_vec-theta) < theta_res) & (np.abs(p_ref_vec-phi) < phi_res)
        labels = label_vec[np.where(candidate_mask == 1)]

        label_i = np.unique(labels)
        label_i = label_i[np.where(label_i > 0)[0]]
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

        # counter += 1
        # print(counter)

# iterate through and compile a list of all labels that were second
overlap_flag_vec = np.asarray(overlap_flag_vec)
shadow_label_list = []
for i in range(len(radial_dist_list)):
    lbs = intersected_label_list[i]
    rs = radial_dist_list[i]
    if len(lbs) > 1:
        si = np.argsort(rs)
        rmin = np.min(rs)
        for s in si[1:]:
            if rs[s] > rmin + 5:
                shadow_label_list.append(lbs[s])

shadow_label_vec = np.unique(np.asarray(shadow_label_list))

print("cleaning masks...")
prob_mask_u = prob_mask_er.copy().astype(int)
for sh in shadow_label_vec:
    prob_mask_u[label_image == sh] = 2
# convert centroids and mask coordinates to spherical coor
# r_image = np.sqrt((z_ref-center_point[0])**2 + (y_ref-center_point[1])**2 + (x_ref-center_point[2])**2)

# use distance transform to select pixels to remove
# pb_ds = np.round(resize(prob_mask_u, (prob_mask_u.shape[0] // 2, prob_mask_u.shape[1] // 2, prob_mask_u.shape[2] // 2), order=0, preserve_range=True)).astype(int)
dist_remove = distance_transform_edt(prob_mask_u != 2)
dist_keep = distance_transform_edt(prob_mask_u != 1)

# raw_indices = np.where(prob_mask.ravel() == 1)[0]
# dist_rm_vec = dist_remove[raw_indices]
# dist_keep_vec = dist_remove[raw_indices]
# rm_indices = raw_indices[dist_rm_vec <= dist_keep_vec]
rm_array = prob_mask & (dist_remove <= dist_keep)

# generate final output
prob_mask_final = prob_mask.copy()
prob_mask_final[rm_array] = 0
label_image_final = label(prob_mask_final)

viewer = napari.view_image(raw_image)
viewer.add_labels(prob_mask_u)
viewer.add_labels(label_image_final)
print("check")
# viewer.add_image(r_image, colormap="magma", opacity=0.4)


if __name__ == '__main__':
    napari.run()