import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.killi_stats.field_extraction import project_fields_to_sphere
from pathlib import Path

if __name__ == '__main__':
    project_list = ["20250621", "20250716", "20250731"]
    well_list = [[1,2,3,4,5,6,7,8,9,10,11,12,15], [0, 1,2,3,4,6,7,8,10,11], [0, 4, 7, 9, 10, 11, 15, 17]]
    # well_list = [[0, 4, 7, 9, 10, 11, 15, 17]]
    sphere_fit_channels = [1, 1, 0]
    # root = Path("/net/trapnell/vol1/home/nlammers/projects/data/killi_tracker/")
    root = Path("/media/nick/cluster/projects/data/killi_tracker/")
    R_um = 550

    for p, proj in enumerate(project_list):
        project_fields_to_sphere(root,
                          proj,
                          wells=well_list[p],
                          R_um=R_um,
                          dist_thresh=50,
                          n_jobs=1,
                          sphere_fit_channel=sphere_fit_channels[p],
                          proj_mode="max")