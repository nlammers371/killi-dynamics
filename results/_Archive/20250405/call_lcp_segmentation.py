from src.build_killi.build_utils import marker_mask_wrapper



if __name__ == "__main__":

    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20250311_LCP1-NLSMSC"

    marker_mask_wrapper(root, project_name=project_name, fluo_channel=0, par_flag=True, mask_range=[1200, 2339])
