from src.build_killi.run02_segment_nuclei import calculate_li_trend, estimate_li_thresh


if __name__ == '__main__':

    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    side1_project_name = "20251126_LCP1-NLSMSC"
    estimate_li_thresh(root, side1_project_name, interval=10, nuclear_channel=None, last_i=None, timeout=60 * 10)

    # get trend
    project_prefix = "20251126_LCP1-NLSMSC"
    calculate_li_trend(root, project_prefix=project_prefix)