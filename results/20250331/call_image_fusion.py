from src.build_killi.build_utils import image_fusion_wrapper

if __name__ == "__main__":
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    out_root = "D:\\Nick\\killi_tracker\\"
    project_name = "20250311_LCP1-NLSMSC"
    image_fusion_wrapper(root, project_name, out_root=out_root, par_flag=False, overwrite=True)