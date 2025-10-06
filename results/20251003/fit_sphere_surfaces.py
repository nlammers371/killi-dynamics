import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.build_yx1.fit_embryo_surface import sphere_fit_wrapper
from pathlib import Path

if __name__ == '__main__':
    project_list = ["20250716", "20250731"]
    # root = Path("/net/trapnell/vol1/home/nlammers/projects/data/killi_tracker/")
    model_name = "tdTom-bright-log-v5"
    root = Path("/media/nick/cluster/projects/data/killi_tracker/")

    for p, proj in enumerate(project_list):
        sphere_fit_wrapper(root,
                          proj,
                          model_name=model_name,
                          n_jobs=4)