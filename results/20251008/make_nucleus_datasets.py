from src.calculate_cell_fields.build_nucleus_data import build_nucleus_df_wrapper
from pathlib import Path


if __name__ == '__main__':

    root = Path("/media/nick/cluster/projects/data/killi_tracker/")
    project_list = ["20250621", "20250716", "20250731"]
    well_list = [None, None, None]
    fluor_channel = [0, 0, None]
    dt_vec = [47/60, 73/60, 68/60]
    model_name = ["tdTom-bright-log-v5", "tdTom-bright-log-v5", "tdTom-bright-log-v5"]
    dt_target = 0.75
    overwrite = True
    n_workers = 8

    for p, proj in enumerate(project_list):
        build_nucleus_df_wrapper(
            root=root,
            project_name=proj,
            wells=well_list[p],
            model_name=model_name[p],
            fluor_channel=fluor_channel[p],
            dT=dt_vec[p],
            n_jobs=n_workers
        )