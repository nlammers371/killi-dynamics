from ultrack.imgproc.flow import timelapse_flow
from src.registration.virtual_fusion import VirtualFuseArray
from src.data_io.zarr_io import open_experiment_array
from pathlib import Path


def infer_optical_flow( root: Path,
                        project: str
                        ):

    vf, _store_path, _resolved_side = open_experiment_array(root,
                                                            project,
                                                            side="virtual_fused",
                                                            verbose=True,
                                                            use_gpu=True,
                                                            interp="linear")

    out_path = root / "optical_flow" / f"{project}_optical_flow.zarr"
    test = vf[:2]
    # compute optical flow
    flow = timelapse_flow(test, store_or_path=out_path, lr=1e-2, num_iterations=2_000)

    return flow

if __name__ == "__main__":
    # --- USER CONFIG ---
    data_root = Path(r"Y:\killi_dynamics")
    project_name = "20251019_BC1-NLS_52-80hpf"
    use_gpu = False
    # -------------------

    flow = infer_optical_flow(root=data_root,
                              project=project_name)