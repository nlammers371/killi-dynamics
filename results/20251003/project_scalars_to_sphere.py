import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build_yx1.project_scalar_fields import field_projection_wrapper
from pathlib import Path

if __name__ == '__main__':
    project_list = ["20250621", "20250716", "20250731"]
    # root = Path("/net/trapnell/vol1/home/nlammers/projects/data/killi_tracker/")
    root = Path("/media/nick/cluster/projects/data/killi_tracker/")
    mode = "max"
    dist_thresh = 50

    for p, proj in enumerate(project_list):
        field_projection_wrapper(root,
                                 proj,
                                 proj_mode=mode,
                                 nside=128,
                                 dist_thresh=50,
                                 n_jobs=4)