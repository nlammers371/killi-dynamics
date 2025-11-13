import sys
from pathlib import Path
# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from src.fluorescence.extract_image_foreground import extract_foreground_intensities
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # root = Path(r"Y:\killi_dynamics")

    project_name = "20251019_BC1-NLS_52-80hpf"
    root = Path("/media/nick/hdd011/killi_dynamics/")
    # call function
    extract_foreground_intensities(root, project_name, n_workers=4, overwrite=False)