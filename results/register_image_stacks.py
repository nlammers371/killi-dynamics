import numpy as np
from src.nucleus_dynamics.utilities.register_image_stacks import registration_wrapper



if __name__ == '__main__':
    experiment_date = "20240611_NLS-Kikume_24hpf_side2"
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    registration_wrapper(root, experiment_date)

    print("Phew")