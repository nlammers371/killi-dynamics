import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build_yx1.project_density_fields import density_projection_wrapper
from pathlib import Path

if __name__ == '__main__':
    project_list = ["20250731"]
    # root = Path("/net/trapnell/vol1/home/nlammers/projects/data/killi_tracker/")
    model_name = "tdTom-bright-log-v5"
    root = Path("/media/nick/cluster/projects/data/killi_tracker/")
    nside = 128
    dist_thresh = 50.0

    for p, proj in enumerate(project_list):
        density_projection_wrapper(root,
                                   project_name=proj,
                                   nside=nside,
                                   dist_thresh=dist_thresh ,
                                   model_name=model_name,
                                   n_jobs=1)