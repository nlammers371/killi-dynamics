import sys
sys.path.append("/functions")
sys.path.append("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")
sys.path.append("E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\morphseq\\")
# sys.path.append("../src/")

from src.vae.auxiliary_scripts.dataset_utils import load_split_train_test
import os
from src.vae.models import VAE, VAEConfig, MetricVAE, MetricVAEConfig, SeqVAEConfig, SeqVAE
from src.vae.auxiliary_scripts.custom_networks import Encoder_Conv_VAE, Decoder_Conv_VAE
from src.vae.trainers import BaseTrainerConfig
from src.vae.pipelines.training import TrainingPipeline

def train_vae(root, train_folder, tracking_model, n_epochs, model_type, input_dim=None, train_suffix='', **kwargs):

    training_keys = ["batch_size", "learning_rate", "n_load_workers"] # optional training config kywords
    # model_keys = ["n_latent", "n_out_channels", "zn_frac", "depth", "nt_xent_temperature"]
    training_args = dict({})
    model_args = dict({})
    for key, value in kwargs.items():
        if key in training_keys:
            if key == "batch_size":
                training_args["per_device_train_batch_size"] = value
                training_args["per_device_eval_batch_size"] = value
            elif key == "n_load_workers":
                training_args["train_dataloader_num_workers"] = value
                training_args["eval_dataloader_num_workers"] = value
            else:
                training_args[key] = value
        else:
            model_args[key] = value

    if input_dim == None:
        input_dim = (1, 96, 96)

    train_dir = os.path.join(root, "shape_images", train_folder, tracking_model)
    # metadata_path = os.path.join(root, "metadata", '')

    if model_type == "MetricVAE":
        # initialize model configuration
        model_config = MetricVAEConfig(
            input_dim=input_dim,
            **model_args
        )

    elif model_type == "VAE":
        # load standard VAE config
        model_config = VAEConfig(
            input_dim=input_dim,
            **model_args
        )

    elif model_type == "SeqVAE":
        # initialize model configuration
        model_config = SeqVAEConfig(
            input_dim=input_dim,
            data_root=root,
            train_folder=train_folder,
            **model_args
        )
        # initialize reference dataset
        print("Making lookup dictionary for sequential pairs...")
        model_config.make_dataset()

    else:
        raise Exception("Unrecognized model type: " + model_type)


    # make output directory to save training results
    if train_suffix == '':
        model_name = model_type + f'_z{model_config.latent_dim:02}_' + f'ne{n_epochs:03}' #+ f'gamma{int(model_config.gamme):04}_' + f'temp{int(model_config.temperature):04}'
    else:
        model_name = model_type + f'_z{model_config.latent_dim:02}_' + f'ne{n_epochs:03}_' + train_suffix #+ f'gamma{int(model_config.gamma):04}_' + f'temp{int(model_config.temperature):04}'  + '_'
    output_dir = os.path.join(root, "shape_models", train_folder, tracking_model, model_name)
    if not os.path.isdir((output_dir)):
        os.makedirs(output_dir)

    # initialize training configuration
    config = BaseTrainerConfig(
        output_dir=output_dir,
        num_epochs=n_epochs,
        **training_args
    )

    # Initialize encoder and decoder
    encoder = Encoder_Conv_VAE(model_config)  # these are custom classes I wrote for this use case
    decoder = Decoder_Conv_VAE(encoder)

    # initialize model
    if model_type == "MetricVAE":
        model = MetricVAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder
        )
    elif model_type == "VAE":
        model = VAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder
        )
    elif model_type == "SeqVAE":
        model = SeqVAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder
        )
    else:
        raise Exception("Unrecognized model type: " + model_type)

    pipeline = TrainingPipeline(
        training_config=config,
        model=model
    )

    train_dataset, eval_dataset = load_split_train_test(train_dir, model_type, model_config, config)
    pipeline(
        train_data=train_dataset,  # here we use the custom train dataset
        eval_data=eval_dataset  # here we use the custom eval dataset
    )

    return output_dir


if __name__ == "__main__":
    # from functions.pythae_utils import *

    #####################
    # Required arguments
    root = "E:\\Nick\\Dropbox (Cole Trapnell's Lab)\\Nick\\killi_tracker\\built_data\\"
    train_folder = "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw"
    train_dir = os.path.join(root, "built_data", "shape_images", train_folder)
    tracking_model = "tracking_v17"
    model_type = "VAE"

    #####################
    # Optional arguments
    train_suffix = "image_high_contrast"
    temperature = 0.01
    batch_size = 64
    n_epochs = 100
    latent_dim = 10
    n_conv_layers = 3
    distance_metric = "euclidean"

    output_dir = train_vae(root, train_folder, tracking_model, train_suffix=train_suffix, model_type=model_type,
                           latent_dim=latent_dim, batch_size=batch_size,
                           n_epochs=n_epochs, temperature=temperature, learning_rate=1e-4, n_conv_layers=n_conv_layers)




