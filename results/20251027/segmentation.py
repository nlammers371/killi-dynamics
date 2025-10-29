from src.segmentation.li_thresholding import run_li_threshold_pipeline

if __name__ == '__main__':
    data_root = r"Y:\killi_dynamics"
    project_name = "20251019_BC1-NLS_52-80hpf"

    run_li_threshold_pipeline(
        root=data_root,
        project_name=project_name)