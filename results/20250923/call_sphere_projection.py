from src.killi_stats.field_extraction import process_dataset
from pathlib import Path

if __name__ == '__main__':
    project_list = ["20250621", "20250716", "20250731"]
    # well_list = [[1,2,3,4,5,6,7,8,9,10,11,12,15], [0, 1,2,3,4,6,7,8,10,11], [0, 4, 7, 9, 10, 11, 15, 17]]
    well_list = [[0, 4, 7, 9, 10, 11, 15, 17]]
    root = Path("/media/nick/cluster/projects/data/killi_tracker/")
    R_um = 550

    for p, proj in enumerate(project_list[2:]):
        process_dataset(root,
                          proj,
                          wells=well_list[p],
                          R_um=R_um,
                          dist_thresh=75,
                          n_jobs=1)