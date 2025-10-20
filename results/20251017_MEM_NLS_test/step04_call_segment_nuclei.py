from src.segmentation import segment_nuclei_thresh
from src.qc import mask_qc_wrapper
from multiprocessing import freeze_support
from pathlib import Path

if __name__ == '__main__':
    freeze_support()
    # load zarr image file
    root = Path(r"E:\pipeline_dev\killi_dynamics")
    project_name = "MEM_NLS_test"
    segment_nuclei_thresh(root, project_name, overwrite=True, n_workers=8)

    mask_qc_wrapper(root, project_name, mask_type="li_segmentation", n_workers=1, overwrite=True)