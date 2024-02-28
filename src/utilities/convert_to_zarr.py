import os
import zarr
import skimage.io as io
from tqdm import tqdm

def convert_tiff_to_zarr(input_folder, output_file):
    # Get list of TIFF files
    tiff_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') or f.endswith('.tiff')]

    if not tiff_files:
        print("No TIFF files found in the directory.")
        return

    # Load the first TIFF file to get shape and dtype
    sample_tiff = io.imread(os.path.join(input_folder, tiff_files[0]))
    shape = sample_tiff.shape
    dtype = sample_tiff.dtype

    # Create Zarr array
    z = zarr.open(output_file, mode='w', shape=(len(tiff_files),) + shape, dtype=dtype, chunks=(1,) + shape)

    # Iterate over TIFF files, read them, and store in Zarr array
    for i, tiff_file in enumerate(tqdm(tiff_files)):
        # print(f"Processing {tiff_file}...")
        tiff_data = io.imread(os.path.join(input_folder, tiff_file))
        z[i] = tiff_data

    print("Conversion completed.")


if __name__ == '__main__':
    # Example usage
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\"
    project_name = "231016_EXP40_LCP1_UVB_300mJ_WT_Timelapse_Raw" #"240219_LCP1_93hpf_to_127hpf"
    image_folder = os.path.join(root, project_name, "")
    output_zarr_file = os.path.join(root, "exported_image_files", project_name + ".zarr")
    if not os.path.isdir(output_zarr_file):
        os.makedirs(output_zarr_file)

    convert_tiff_to_zarr(image_folder, output_zarr_file)