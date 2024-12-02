from src.build_killi.segment_nuclei import segment_nuclei


if __name__ == '__main__':

    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20241114_LCP1-NLSMSC_side2"
    segment_nuclei(root, project_name, last_i=805)