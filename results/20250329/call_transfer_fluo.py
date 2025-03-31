from src.build_killi.build_utils import transfer_fluorescence_wrapper

if __name__ == '__main__':
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    # project name
    project_name = "20250311_LCP1-NLSMSC"
    tracking_config = "tracking_20250328_redux"  # only used for ML segmentation
    tracking_range = [0, 2339]
    suffix = "_cb"

    # transfer fluorescence
    transfer_fluorescence_wrapper(root, project_name=project_name, tracking_config=tracking_config,
                                  tracking_range=tracking_range, par_flag=True, n_workers=24, suffix=suffix)

    print("Done transferring fluorescence.")