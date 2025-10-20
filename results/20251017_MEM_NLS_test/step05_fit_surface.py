from src.geometry.geom_wrappers import fit_surf_sphere_trend, fit_sh_trend
from pathlib import Path


if __name__ == '__main__':

    root = Path(r"E:\pipeline_dev\killi_dynamics")
    pyproject_name = "MEM_NLS_test"
    seg_type = "li_segmentation"

    # sphere_df = fit_surf_sphere_trend(root=root,
    #                                   project_name=pyproject_name,
    #                                   seg_type="li_segmentation")

    fit_sh_trend(root=root,
                 project_name=pyproject_name,
                 seg_type="li_segmentation")
