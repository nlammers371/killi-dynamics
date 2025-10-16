import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

import multiprocessing
from src.build_yx1.export_nd2_to_zarr import export_nd2_to_zarr


def main():
    experiment_date_vec = ["20250731", "20250621", "20250716"]  #, "20250530", "20250425"]
    # root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/"
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/pecfin_dynamics/"
    root = "/media/nick/cluster/projects/data/killi_tracker/"
    # root = "/net/trapnell/vol1/home/nlammers/projects/data/killi_tracker/"
    overwrite_flag = True
    nuclear_channel_vec = [0, 1, 1]
    channel_names_vec = [["NLS-mScarlet"], ["NLS-mScarlet", "BC1"], ["NLS-mScarlet", "BC1"]]  #, ["tbx5a-StayGold", "H2B-tdTom"]]

    for e, experiment_date in enumerate(experiment_date_vec):

        print("#########################")
        print("Exporting nucleus data for experiment {}".format(experiment_date))
        print("#########################")

        nuclear_channel = nuclear_channel_vec[e]
        channel_names = channel_names_vec[e]
        export_nd2_to_zarr(root, experiment_date, overwrite_flag, nuclear_channel=nuclear_channel,
                           channel_names=channel_names, save_z_projections=True, num_workers=8)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()