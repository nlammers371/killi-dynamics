from src.build_killi.fuse_masks import fuse_wrapper
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    # side 1
    project_name = "20250311_LCP1-NLSMSC"
    fuse_wrapper(root, project_prefix=project_name, par_flag=False, overwrite=False)
