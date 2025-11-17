import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

from ultrack.imgproc.flow import timelapse_flow
from src.registration.virtual_fusion import VirtualFuseArray
from src.data_io.zarr_io import open_experiment_array
from pathlib import Path


def infer_optical_flow( root: Path,
                        project: str,
                        overwrite: bool = False):

    vf, _store_path, _resolved_side = open_experiment_array(root,
                                                            project,
                                                            side="virtual_fused",
                                                            verbose=True,
                                                            use_gpu=False, # True fails for multiple timepoints (which is weird)
                                                            interp="nearest")

    out_path = root / "optical_flow" / f"{project}_optical_flow.zarr"
    if out_path.exists() and overwrite:
        shutil.rmtree(out_path)
    elif  out_path.exists():
        print(f"Optical flow already exists at {out_path}. Skipping computation.")
        return

    print("Loading virtual fusion array...")


    # compute optical flow
    print("Computing optical flow...")
    flow = timelapse_flow(vf, channel_axis=0, store_or_path=out_path, lr=1e-2, num_iterations=1_000)

    return flow


if __name__ == "__main__":
    # --- USER CONFIG ---
    # data_root = Path(r"Y:\killi_dynamics")
    data_root = Path("/media/nick/hdd011/killi_dynamics/")
    project_name = "20251019_BC1-NLS_52-80hpf"
    # -------------------

    flow = infer_optical_flow(root=data_root,
                              project=project_name,
                              overwrite=True)