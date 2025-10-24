from src.data_io.zarr_utils import merge_zarr_sides, convert_shift_table_to_transform_attrs
from pathlib import Path
from typing import Optional

if __name__ == "__main__":
    side1_path = Path(r"E:/Nick/killi_dynamics/built_data/zarr_image_files/20250419_BC1-NLSMSC_side1.zarr")
    side2_path = Path(r"E:/Nick/killi_dynamics/built_data/zarr_image_files/20250419_BC1-NLSMSC_side2.zarr")
    out_store_path = Path(r"E:/Nick/killi_dynamics/built_data/zarr_image_files/20250419_BC1-NLSMSC.zarr")
    project_name = "20250419_BC1-NLSMSC"
    merge_zarr_sides(
                                        side1_path=side1_path,
                                        side2_path=side2_path,
                                        out_store_path=out_store_path,
                                        project_name=project_name,
                                        overwrite=True,
                                        )

    shift_table_path = Path(r"E:\Nick\killi_dynamics\metadata\legacy\20250419_BC1-NLSMSC_side1\20250419_BC1-NLSMSC_side2_to_20250419_BC1-NLSMSC_side1_shift_df.csv")
    convert_shift_table_to_transform_attrs(
                                        fused_store_path=out_store_path,
                                        shift_csv_path=shift_table_path)