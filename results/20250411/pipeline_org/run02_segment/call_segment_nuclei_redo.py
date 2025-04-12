from src.build_killi.run02_segment_nuclei import segment_nuclei
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    # side 1
    project_name = "20250311_LCP1-NLSMSC_side1"
    segment_nuclei(root, project_name, overwrite=True, par_flag=True)

    # side 2
    project_name = "20250311_LCP1-NLSMSC_side2"
    segment_nuclei(root, project_name, overwrite=True, par_flag=True)