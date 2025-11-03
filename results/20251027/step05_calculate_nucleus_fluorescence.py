from src.fluorescence.extract_image_foreground import extract_foreground_intensities
from pathlib import Path
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

    root = Path(r"Y:\killi_dynamics")
    project_name = "20251019_BC1-NLS_52-80hpf"

    # call function
    extract_foreground_intensities(root, project_name, n_workers=12, overwrite=True)