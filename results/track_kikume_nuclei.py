import numpy as np
from src.nucleus_dynamics.tracking.perform_tracking import perform_tracking


if __name__ == '__main__':
    experiment_date = "20240611_NLS-Kikume_24hpf_side2"
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    tracking_config = "tracking_jordao_20240918.txt"
    segmentation_model = "LCP-Multiset-v1"
    start_i = 0
    stop_i = 1600
    perform_tracking(root, experiment_date, tracking_config, seg_model=segmentation_model, start_i=start_i, stop_i=stop_i)