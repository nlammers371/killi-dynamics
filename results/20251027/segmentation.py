from src.segmentation.li_thresholding import run_li_threshold_pipeline
from src.segmentation.segmentation_wrappers import segment_nuclei_thresh
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

if __name__ == '__main__':
    data_root = r"Y:\killi_dynamics"
    project_name = "20251019_BC1-NLS_52-80hpf"

    # run_li_threshold_pipeline(
    #     root=data_root,
    #     li_tol=.01,
    #     use_subsampling=True,
    #     project_name=project_name)

    segment_nuclei_thresh(
                    root=data_root,
                    project_name=project_name,
                    nuclear_channel=1,
                    segment_sides_separately=True,
                    n_workers=1,
                    n_thresh=3,
                    overwrite=False,
                    last_i=None,
                    preproc_flag=True,
                    thresh_factors=None,
                )
