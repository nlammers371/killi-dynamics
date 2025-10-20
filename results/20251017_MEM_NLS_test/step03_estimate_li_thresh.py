from src.segmentation import calculate_li_trend, estimate_li_thresh
from pathlib import Path

if __name__ == '__main__':

    # load zarr image file
    root = Path(r"E:\pipeline_dev\killi_dynamics")
    project_prefix = "MEM_NLS_test"
    estimate_li_thresh(root, project_prefix, interval=3, nuclear_channel=None, last_i=None, timeout=60 * 10)

    # get trend
    calculate_li_trend(root, project_prefix=project_prefix)