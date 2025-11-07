from src.geometry.geom_wrappers import fit_surf_sphere_trend, fit_sh_trend
from pathlib import Path
import logging

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,  # or WARNING to silence most messages
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    root = r"Y:\killi_dynamics"
    project_name = "20251019_BC1-NLS_52-80hpf"
    seg_type = "li_segmentation"

    sphere_df = fit_surf_sphere_trend(root=root,
                                      overwrite=True,
                                      project_name=project_name,
                                      seg_type="li_segmentation",
                                      n_workers=12)

    # fit_sh_trend(root=root,
    #              project_name=project_name,
    #              seg_type="li_segmentation",
    #              n_workers=12,)
