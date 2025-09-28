from src.nucleus_dynamics.build.build01_segment_nuclei_zarr import cellpose_segmentation


if __name__ == '__main__':

    # load zarr image file
    root = "/media/nick/cluster/projects/data/killi_tracker/"
    model_path = "/media/nick/cluster/projects/data/pecfin_dynamics/built_data/cellpose_training/standard_models/tdTom-bright-log-v5"
    project_name = "20250716"
    cellpose_segmentation(root=root,
                          experiment_date=project_name,
                          pretrained_model=model_path,
                          nuclear_channel=1)