from src.build_killi.process_masks import mask_qc_wrapper
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    # side 1
    # project_name = "20250311_LCP1-NLSMSC_side1"
    # mask_qc_wrapper(root, project_name, par_flag=False, overwrite=False, n_workers=8)

    # side 2
    project_name = "20250311_LCP1-NLSMSC_side2"
    mask_qc_wrapper(root, project_name, par_flag=True, overwrite=True)