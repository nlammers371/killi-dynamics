from src.segmentation.li_thresholding import run_li_threshold_pipeline
from src.segmentation.segmentation_wrappers import segment_nuclei_thresh
from src.qc.mask_qc import mask_qc_wrapper
from src.registration.virtual_fusion import VirtualFuseArray
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

if __name__ == '__main__':
    root = Path(r"Y:\killi_dynamics")
    project_name = "20251019_BC1-NLS_52-80hpf"

    # run_li_threshold_pipeline(
    #     root=data_root,
    #     li_tol=.01,
    #     use_subsampling=True,
    #     project_name=project_name)

    # segment_nuclei_thresh(
    #                 root=data_root,
    #                 project_name=project_name,
    #                 nuclear_channel=1,
    #                 segment_sides_separately=True,
    #                 n_workers=12,
    #                 n_thresh=3,
    #                 overwrite=False,
    #                 last_i=None,
    #                 preproc_flag=True,
    #                 thresh_factors=None,
    #             )

    # mask_qc_wrapper(root=data_root,
    #                 project=project_name,
    #                 mask_type="li_segmentation",
    #                 n_workers=12,
    #                 overwrite=True)

    # fuse masks
    store_path = root / "segmentation" / "li_segmentation" / f"{project_name}_masks.zarr"
    vf = VirtualFuseArray(
        store_path=store_path,
        is_mask=True,
        subgroup_key="clean",
        use_gpu=False,
    )

    vf.write_fused(
        subgroup="clean",  # writes to fused/clean
        overwrite=True,
        n_workers=12
    )