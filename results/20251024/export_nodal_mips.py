import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

import multiprocessing
from src.export.nd2_export import export_nd2_to_zarr


def main():
    experiment_date_vec = ["20251010"]  #, "20250530", "20250425"]
    # root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/pecfin_dynamics/"
    root = r"E:Nick/symmetry_breaking/"
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/killi_tracker/"
    overwrite_flag = True
    nuclear_channel_vec = [0]
    channel_names_vec = [["DAPI"]]  #, ["tbx5a-StayGold", "H2B-tdTom"]]

    for e, experiment_date in enumerate(experiment_date_vec):

        print("#########################")
        print("Exporting nucleus data for experiment {}".format(experiment_date))
        print("#########################")

        nuclear_channel = nuclear_channel_vec[e]
        channel_names = channel_names_vec[e]
        export_nd2_to_zarr(root, experiment_date, overwrite_flag, nuclear_channel=nuclear_channel,
                           channel_names=channel_names, save_z_projections=True, num_workers=1)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()