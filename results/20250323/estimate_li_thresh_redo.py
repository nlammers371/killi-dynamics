from src.segmentation import segment_nuclei, estimate_li_thresh, calculate_li_trend


if __name__ == '__main__':

    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    # side1_project_name = "20250311_LCP1-NLSMSC_side1"
    # # # segment_nuclei(root, project_name)
    # estimate_li_thresh(root, side1_project_name, interval=50, nuclear_channel=None, last_i=None, timeout=60 * 10)
    #
    # side2_project_name = "20250311_LCP1-NLSMSC_side2"
    # # segment_nuclei(root, project_name)
    # estimate_li_thresh(root, side2_project_name, interval=50, nuclear_channel=None, start_i=500, last_i=None, timeout=60 * 10)

    li_df_full = calculate_li_trend(root=root, project_prefix="20250311_LCP1-NLSMSC")