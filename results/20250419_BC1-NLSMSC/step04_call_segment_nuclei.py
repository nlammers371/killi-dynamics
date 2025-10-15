from src.segmentation import segment_nuclei
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    # side 1
    side1_project_name = "20250419_BC1-NLSMSC_side1"
    segment_nuclei(root, side1_project_name, overwrite=False, par_flag=True)

    # side 2
    side2_project_name = "20250419_BC1-NLSMSC_side2"
    segment_nuclei(root, side2_project_name, overwrite=False, par_flag=True)