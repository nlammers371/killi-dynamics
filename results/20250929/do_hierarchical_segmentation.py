from src.nucleus_dynamics.build.build02_stitch_nuclear_masks import stitch_cellpose_labels
import numpy as np

# set read/write paths
# root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
root = "/media/nick/cluster/projects/data/killi_tracker/"
experiment_date_vec = ["20250716"]
pretrained_model_vec = ["tdTom-bright-log-v5"]  #[pretrained_model0, pretrained_model1, pretrained_model1, pretrained_model0, pretrained_model0]
overwrite = True
prob_thresh_range = np.arange(1, 10, 3)
well_range = np.arange(0, 1)
seg_res = 0.65

for e, experiment_date in enumerate(experiment_date_vec):

    model_name = pretrained_model_vec[e]

    stitch_cellpose_labels(root=root, model_name=model_name, prob_thresh_range=prob_thresh_range, well_range=well_range,
                                      experiment_date=experiment_date, overwrite=overwrite, seg_res=seg_res)