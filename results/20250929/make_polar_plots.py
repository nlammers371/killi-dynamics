from src.killi_stats.make_field_movies import healpix_to_equalarea_mp4
from pathlib import Path

if __name__ == '__main__':
    project_name = "20250716"
    root = "/media/nick/cluster/projects/data/killi_tracker/"
    out_dir = "/media/nick/cluster/projects/data/killi_tracker/output/polar_movies/"
    # well_list = [0, 4, 7, 9, 10, 11, 15, 17]
    R_um = 550
    fps = 10

    projection_path = Path(root) / "output_data" / "sphere_projections" / project_name
    well_list = sorted(projection_path.glob("*.zarr"))
    for w, well_path in enumerate(well_list):
        healpix_to_equalarea_mp4(
            well_zarr_path=well_path,
            out_mp4="well0003.mp4",
            values_key="raw",
            channel=1,  # or 0
            bins=128,
            fps=10,
            phi0_deg=0.0,
            cmap="viridis",
        )