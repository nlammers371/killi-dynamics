from src._Archive.image_utils.do_mip_projections import calculate_mip_projection
from dask.distributed import Client, LocalCluster


if __name__ == "__main__":

    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20241126_LCP1-NLSMSC_v0"

    cluster = LocalCluster(n_workers=12, threads_per_worker=1)
    client = Client(cluster)

    # DO MIP PROJECTIONS
    calculate_mip_projection(root, project_name, dual_sided=False)