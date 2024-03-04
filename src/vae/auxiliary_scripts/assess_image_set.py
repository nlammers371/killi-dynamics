import glob as glob
from src.vae.auxiliary_scripts.dataset_utils import *
import os
import skimage.io as io
from skimage.transform import rescale
from src.vae.models.auto_model import AutoModel
import matplotlib.pyplot as plt
import umap.umap_ as umap
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import pandas as pd
from pythae.data.datasets import collate_dataset_output
from tqdm import tqdm
# from pythae.trainers.base_trainer_verbose import base_trainer_verbose
#from src.functions.dataset_utils import ContrastiveLearningDataset, ContrastiveLearningViewGenerator
from src.utilities.functions import path_leaf
from torch.utils.data import DataLoader
import json
from typing import Any, Dict, List, Optional, Union
import ntpath
from src.vae.auxiliary_scripts.assess_vae_results import calculate_UMAPs


def set_inputs_to_device(device, inputs: Dict[str, Any]):
    inputs_on_device = inputs

    if device == "cuda":
        cuda_inputs = dict.fromkeys(inputs)

        for key in inputs.keys():
            if torch.is_tensor(inputs[key]):
                cuda_inputs[key] = inputs[key].cuda()

            else:
                cuda_inputs[key] = inputs[key]
        inputs_on_device = cuda_inputs

    return inputs_on_device

# Script to generate image reconstructions and latent space projections for a designated set of embryo images
def assess_image_set(image_path, tracking_path, trained_model_path, out_path, n_im_figures=100, rs_factor=1.0, batch_size=64):

    rs_flag = rs_factor != 1
    # check for GPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # load raw metadata DF
    print("Loading metadata...")
    tracks_df = pd.read_csv(os.path.join(tracking_path, "tracks.csv"))

    # load the model
    print("Loading model...")
    trained_model = AutoModel.load_from_folder(os.path.join(trained_model_path, 'final_model'))

    # load images 
    # image_path_list = []
    # for image_prefix in image_prefix_list:
    #     im_list = sorted(glob.glob(os.path.join(image_path, image_prefix + "*")))
    #     image_path_list += im_list
    # 
    # image_snip_names = [path_leaf(im)[:-4] for im in image_path_list]


    # pair down data entries
    latent_shape_df = tracks_df.loc[:, ["track_id", "t"]].copy()

    # initialize new columns
    # latent_shape_df["recon_mse"] = np.nan
    # snip_id_vec = list(latent_shape_df["snip_id"])
    # keep_indices = np.asarray([i for i in range(len(snip_id_vec)) if snip_id_vec[i] in image_snip_names])

    # latent_shape_df = latent_shape_df.loc[keep_indices, :].copy()
    # latent_shape_df.reset_index(inplace=True)

    # load and store the images
    print("Loading images...")
    # input_image_path = os.path.join(out_path, "input_images", "class0")
    # if not os.path.isdir(input_image_path):
    #     os.makedirs(input_image_path)
    # 
    # for i in tqdm(range(len(image_path_list))):
    #     img_raw = io.imread(image_path_list[i])
    #     if len(img_raw.shape) == 3:
    #         img_raw = img_raw[:, :, 0]
    #     
    #     if rs_flag:
    #         img = rescale(img_raw.astype(np.float16), rs_factor, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    #     else:
    #         img = img_raw
    #     im_name = path_leaf(image_path_list[i])
    #     # save
    #     io.imsave(fname=os.path.join(input_image_path, im_name), arr=img)

    data_transform = make_basic_rs_transform()
    dataset = MyCustomDataset(
            root=image_path,
            transform=data_transform,
            return_name=True
        )

    n_images = len(dataset)
    fig_indices = np.random.choice(range(n_images), np.min([n_images, n_im_figures]), replace=False)
    dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_dataset_output,
                )
    # initialize latent variable columns
    new_cols = []
    for n in range(trained_model.latent_dim):
        if (trained_model.model_name == "MetricVAE") or (trained_model.model_name == "SeqVAE"):
            if n in trained_model.nuisance_indices:
                new_cols.append(f"z_mu_n_{n:02}")
                # new_cols.append(f"z_sigma_n_{n:02}")
            else:
                new_cols.append(f"z_mu_b_{n:02}")
                # new_cols.append(f"z_sigma_b_{n:02}")
        else:
            new_cols.append(f"z_mu_{n:02}")
            # new_cols.append(f"z_sigma_{n:02}")
    latent_shape_df.loc[:, new_cols] = np.nan

    # make subdir for images
    image_path = os.path.join(out_path, "images_reconstructions")
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    # make subdir for comparison figures
    recon_fig_path = os.path.join(out_path, "recon_figs")
    if not os.path.isdir(recon_fig_path):
        os.makedirs(recon_fig_path)

    print("Extracting latent space vectors and testing image reconstructions...")
    trained_model = trained_model.to(device)
    # get the dataloader
    iter_i = 0
    for n, inputs in enumerate(tqdm(dataloader)):

        inputs = set_inputs_to_device(device, inputs)
        x = inputs["data"]
        y = inputs["label"]

        encoder_output = trained_model.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)

        z_out, eps = trained_model._sample_gauss(mu, std)


        # encoder_output = encoder_output.detach().cpu()
        # ###
        # # Add recon loss and latent encodings to the dataframe
        # get indices
        fnames = [path_leaf(f) for f in y[0]]
        track_id_vec = np.asarray([int(fn[8:12]) for fn in fnames])
        time_id_vec = np.asarray([int(fn[14:18]) for fn in fnames])
        df_ind_vec = np.asarray([np.where((latent_shape_df["track_id"] == track_id_vec[i]) &
                                          (latent_shape_df["t"] == time_id_vec[i]))[0][0] for i in range(len(track_id_vec))])


        # add latent encodings
        zm_array = np.asarray(encoder_output[0].detach().cpu())
        # recon_x_out = trained_model.decoder(z_out)["reconstruction"]
        for z in range(trained_model.latent_dim):
            if (trained_model.model_name == "MetricVAE") or (trained_model.model_name == "SeqVAE"):
                if z in trained_model.nuisance_indices:
                    latent_shape_df.loc[df_ind_vec, f"z_mu_n_{z:02}"] = zm_array[:, z]
                else:
                    latent_shape_df.loc[df_ind_vec, f"z_mu_b_{z:02}"] = zm_array[:, z]
            else:
                latent_shape_df.loc[df_ind_vec, f"z_mu_{z:02}"] = zm_array[:, z]

        recon_x_out = trained_model.decoder(z_out)["reconstruction"]

        # x = x.detach().cpu()
        recon_x_out = recon_x_out.detach().cpu()

        for b in range(x.shape[0]):

            if iter_i in fig_indices:


                # show results with normal sampler
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

                axes[0].imshow(np.squeeze(np.squeeze(x.detach().cpu()[b, 0, :, :])), cmap='gray')
                axes[0].axis('off')

                axes[1].imshow(np.squeeze(recon_x_out[b, 0, :, :]), cmap='gray')
                axes[1].axis('off')

                plt.tight_layout(pad=0.)

                plt.savefig(
                    os.path.join(recon_fig_path, fnames[b].replace(".jpg", "") + '_loss.jpg'))
                plt.close()

                # save just the recon on its own
                int_recon_out = (np.squeeze(np.asarray(recon_x_out[b, 0, :, :]))*255).astype(np.uint8)
                io.imsave(fname=os.path.join(image_path, fnames[b].replace(".jpg", "") + '_loss.jpg'), arr=int_recon_out)

            iter_i += 1


    # now fit 2 and 3-dimensional UMAP models 
    print("Calculating UMAPS...")
    latent_shape_df = calculate_UMAPs(latent_shape_df)

    print("Saving...")
    latent_shape_df.to_csv(os.path.join(out_path, "latent_shape_df.csv"))
       

if __name__ == "__main__":

    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    im_project_name = "240219_LCP1_93hpf_to_127hpf"
    im_track_name = "tracking_cell"
    image_path = os.path.join(root, "built_data", "shape_images", im_project_name)
    # image_path = os.path.join(root, "training_data", "20231106_ds", "train", "20230525")
    tracking_path = os.path.join(root, "built_data", "tracking", im_project_name, im_track_name)
    out_path = os.path.join(tracking_path, "shape_analysis")
    rs_factor = 1.0
    batch_size = 64  # batch size to use generating latent encodings and image reconstructions

    # get path to model
    train_root = os.path.join(root, "built_data", "shape_models")
    project_name = "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
    tracking_name = "tracking_v17"
    model_name = "VAE_z50_ne250_image_recon_loss_upweight_v2"
    training_instance = "VAE_training_2024-02-29_14-47-14"
    model_dir = os.path.join(train_root, project_name, tracking_name, model_name, training_instance)


    assess_image_set(image_path=image_path, 
                     tracking_path=tracking_path, 
                    trained_model_path=model_dir, 
                    out_path=out_path, 
                    rs_factor=rs_factor, 
                    batch_size=batch_size)