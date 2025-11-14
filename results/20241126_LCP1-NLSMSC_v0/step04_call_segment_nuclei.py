from src.build_killi.run02_segment_nuclei import segment_nuclei
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"

    # side 1
    n_trhresholds = 5
    thresh_factors = [0.75, 1.25]
    project_name = "20241126_LCP1-NLSMSC_v0"
    segment_nuclei(root,
                   project_name,
                   nuclear_channel=1,
                   overwrite=False,
                   par_flag=True,
                   n_workers=8,
                   preproc_flag=True,  # if false tells script NOT to apply LoG and inversion
                   n_thresh=n_trhresholds,
                   thresh_factors=thresh_factors
                   )

    # side 2
    # side2_project_name = "20250419_BC1-NLSMSC_side2"
    # segment_nuclei(root, side2_project_name, overwrite=False, par_flag=True)