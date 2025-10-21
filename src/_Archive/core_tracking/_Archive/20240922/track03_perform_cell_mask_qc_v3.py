import napari
import os

import pandas as pd
import skimage.io as io
from skimage.transform import resize
import glob2 as glob
# from skimage import label
import skimage
from skimage.morphology import ball
import numpy as np
import json
from src._Archive.utilities.functions import sphereFit, cart_to_sphere
from skimage.measure import label, regionprops
from src.utilities.functions import path_leaf
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

def perform_mask_qc(root, project_name, overwrite_flag=False,
                    prob_thresh=-8, cell_thickness=20, min_cell_size=135, n_points_fit=10000):

    metadata_folder = os.path.join(root, "metadata", project_name)
    if not os.path.isdir(metadata_folder):
        os.makedirs(metadata_folder)

    # prob datasets
    label_folder = os.path.join(root, "built_data", "cellpose_output", project_name, "")
    prob_list = sorted(glob.glob(os.path.join(label_folder + "*_probs.tif")))
    
    save_directory = os.path.join(root, "built_data", "cleaned_cell_labels", project_name, "")
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)
        
    print("loading data and making image masks...")
    # load metadata
    metadata_file_path = os.path.join(metadata_folder, "metadata.json")
    f = open(metadata_file_path)
    metadata = json.load(f)
    
    pixel_res_prob = np.asarray([metadata["ProbPhysicalSizeZ"], metadata["ProbPhysicalSizeY"], metadata["ProbPhysicalSizeX"]])
    
    # metadata_out_path = os.path.join(label_folder, "metadata.json")
    # with open(metadata_out_path, 'w') as json_file:
    #     json.dump(metadata, json_file)
    
    # check for pre-existing sphere center dataset
    sphere_df_path = os.path.join(metadata_folder, "sphere_df.csv")
    prev_df_flag = False
    if (not overwrite_flag) and os.path.isfile(sphere_df_path):
        sphere_df_prev = pd.read_csv(sphere_df_path)
        prev_df_flag = True

    # iter_i = 500
    sphere_df_list = []
    for iter_i in tqdm(range(0, len(prob_list))):

        prob_name = path_leaf(prob_list[iter_i])
        outname = prob_name.replace("probs", "labels")

        if (not os.path.isfile(save_directory + outname)) | overwrite_flag:
            # load image
            prob_image = io.imread(prob_list[iter_i])

            if project_name == "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw":
                prob_image = resize(prob_image, (prob_image.shape[0] // 2, prob_image.shape[1] // 2, prob_image.shape[2] // 2), preserve_range=True)

            # raw_image = resize(raw_image, prob_image.shape, preserve_range=True)
            prob_mask = prob_image >= prob_thresh
            # prob_mask_strict = prob_image >= 0

            # label_image_out = label(prob_mask)
            label_image = label(prob_mask)  # use this version to filter

            # generate reference grid
            print("calculating best-fit sphere...")
            z_ref, y_ref, x_ref = np.meshgrid(range(prob_mask.shape[0]),
                                              range(prob_mask.shape[1]),
                                              range(prob_mask.shape[2]),
                                              indexing="ij")

            z_ref = z_ref*pixel_res_prob[0]
            y_ref = y_ref*pixel_res_prob[1]
            x_ref = x_ref*pixel_res_prob[2]

            # get location of masked pixels
            nc_z = z_ref[prob_mask]
            nc_y = y_ref[prob_mask]
            nc_x = x_ref[prob_mask]

            fit_indices = np.random.choice(range(len(nc_z)), np.min([n_points_fit, len(nc_z)]), replace=False)
            # fit to sphere
            center_point = np.empty((3, ))
            radius, center_point[2], center_point[1], center_point[0] = sphereFit(nc_x[fit_indices], nc_y[fit_indices], nc_z[fit_indices])
            sphere_df = pd.DataFrame(center_point[np.newaxis, :], columns=["Z", "Y", "X"])
            sphere_df["r"] = radius
            sphere_df["t"] = iter_i
            sphere_df["project"] = project_name
            sphere_df_list.append(sphere_df)

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
            index_vec = np.where(prob_mask.ravel())[0]
            label_vec = label_image[prob_mask]

            r_min = np.percentile(r_ref_vec, 10)
            r_max = np.ceil(r_min + 2*cell_thickness)

            # define bounds
            grid_res = 25
            # r_grid = np.linspace(np.min(r_ref), np.max(r_ref), grid_res)
            phi_grid = np.linspace(np.min(p_ref), np.max(p_ref), grid_res)
            theta_grid = np.linspace(np.min(t_ref), np.max(t_ref), grid_res)

            theta_res = np.abs(theta_grid[1]-theta_grid[0]) / 2
            phi_res = np.abs(phi_grid[1]-phi_grid[0]) / 2


            label_index = np.unique(label_image)
            rad_max_vec = np.ones((len(label_index),))*r_max

            for p, phi in enumerate(phi_grid):
                for t, theta in enumerate(theta_grid):
                    # flag masks with nearby centroids
                    candidate_mask = (np.abs(t_ref_vec-theta) < theta_res) & (np.abs(p_ref_vec-phi) < phi_res)
                    candidate_indices = index_vec[candidate_mask]
                    labels = label_vec[np.where(candidate_mask == 1)]
                    lbi = np.unique(labels)
                    r_vals = r_ref_vec[candidate_mask == 1]
                    if len(r_vals) > 1:
                        rmin = np.min([np.min(r_vals), r_max-cell_thickness])
                        rmax = rmin + cell_thickness
                        for lb in lbi:
                            rad_max_vec[lb] = np.min([rmax, rad_max_vec[lb]])


            rm_index_vec = []
            for lb in range(1, len(rad_max_vec)):
                sub_indices = label_vec == lb
                arr_indices = index_vec[sub_indices]
                r_vals = r_ref_vec[sub_indices]
                rm_indices = arr_indices[r_vals > rad_max_vec[lb]]
                rm_index_vec.append(rm_indices)

            rm_index_vec = np.unique(np.concatenate(rm_index_vec))
            rm_sub_vec = np.unravel_index(rm_index_vec, prob_mask.shape)
            print("cleaning masks...")
            prob_mask_clean = prob_mask.copy().astype(int)
            prob_mask_clean[rm_sub_vec] = 0

            label_image_clean = label(prob_mask_clean)
            prob_mask_final = prob_mask_clean.copy()

            # remove small regions
            regions = regionprops(label_image_clean) #, ["centroid", "coords", "area"])
            area_vec = np.asarray([r["area"] for r in regions])*np.prod(pixel_res_prob)
            rm_indices = np.where(area_vec < min_cell_size)[0]
            for ind in rm_indices:
                prob_mask_final[label_image_clean == regions[ind].label] = 0
            label_image_final = label(prob_mask_final).astype(np.uint16)

            # save
            io.imsave(save_directory + outname, label_image_final)

    if prev_df_flag:
        sphere_df_list.append(sphere_df_prev)

    sphere_df_master = pd.concat(sphere_df_list, ignore_index=True)
    sphere_df_master = sphere_df_master.drop_duplicates(subset=["t", "project"])
    sphere_df_master.to_csv(os.path.join(metadata_folder, "sphere_df.csv"), index=False)
    return {}
# viewer = napari.view_image(raw_image)
# viewer.add_labels(prob_mask*2)
# viewer.add_labels(prob_mask_final)

# print("check")


if __name__ == '__main__':
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "240219_LCP1_93hpf_to_127hpf" #"231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"

    perform_mask_qc(root, project_name, min_cell_size=270)