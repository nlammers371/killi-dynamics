from src.data_io.czi_export import export_czi_to_zarr
from pathlib import Path

if __name__ == '__main__':
    raw_data_root = Path(r"D:\Bria\20250611_UTR-NLS_52hpf-76hpf")
    project_name = "20250611_UTR-NLS_52hpf-76hpf"
    save_root = Path(r"Y:\killi_dynamics")
    resampling_scale = [3.0, 0.85, 0.85]
    n_workers = 6
    channel_names = ["UTR-mNeon", "NLS-mScar"] # resample YX to 0.5 um/px
    tres = 120

    export_czi_to_zarr(
        raw_data_dir=raw_data_root,
        project_name=project_name,
        save_root=save_root,
        resampling_scale=resampling_scale,
        n_workers=n_workers,
        channel_names=channel_names,
        tres=tres,
        overwrite_flag=False,
        register_two_sided=True)