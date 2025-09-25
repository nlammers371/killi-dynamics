from src.killi_stats.field_extraction import process_dataset
from pathlib import Path

if __name__ == '__main__':
    project_list = ["20250621", "20250716", "20250731"]
    root = Path("/media/nick/cluster/projects/data/killi_tracker/")
    R_um = 550

    sphere_df, field_db = process_dataset(root, project_list[0], R_um=R_um, n_jobs=8)