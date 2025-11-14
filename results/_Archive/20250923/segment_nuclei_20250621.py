import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.nucleus_dynamics.build.build01_segment_nuclei_zarr import cellpose_segmentation


if __name__ == '__main__':

    # load zarr image file
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/killi_tracker/"
    # model_path = "/net/trapnell/vol1/home/nlammers/projects/data/pecfin_dynamics/built_data/cellpose_training/standard_models/tdTom-bright-log-v5"
    root = "/media/nick/cluster/projects/data/killi_tracker/"
    model_path = "/media/nick/cluster/projects/data/pecfin_dynamics/built_data/cellpose_training/standard_models/tdTom-bright-log-v5"
    project_name = "20250621"
    cellpose_segmentation(root=root,
                          experiment_date=project_name,
                          pretrained_model=model_path,
                          preproc_sigma=[1, 3, 3],
                          well_list=[11], #[6,7,8,9,10,11,12,15], #[1,2,3,4,5,6,7,8,9,10,11,12,15],
                          nuclear_channel=1)