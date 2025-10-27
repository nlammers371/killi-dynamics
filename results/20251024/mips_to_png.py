import zarr
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

if __name__ == "__main__":
    zarr_folder = Path(r"E:\Nick\symmetry_breaking\built_data\zarr_image_files\20251010\mips")
    png_folder = Path(r"E:\Nick\symmetry_breaking\built_data\png_mips\20251010")
    zarr_list = sorted(list(zarr_folder.glob("*_z.zarr")))
    png_folder.mkdir(parents=True, exist_ok=True)

    for zarr_path in tqdm(zarr_list, desc="Converting MIP zarrs to PNGs"):
        print(f"Processing {zarr_path.name}...")
        mip_zarr = zarr.open(zarr_path, mode="r")
        mip_array = np.squeeze(np.array(mip_zarr)).astype(float)

        # --- subtractive Gaussian background removal ---
        # --- subtract background ---
        bg = gaussian_filter(mip_array, sigma=75)
        # mip_array -= bg
        bg[bg == 0] = 1e-6  # avoid division by zero
        mip_array = mip_array / bg
        # mip_array[mip_array < 0] = 0
        # --- clip using more aggressive percentile stretch ---
        mask = mip_array > np.percentile(mip_array, 20)  # ignore dim pixels for scaling
        vals = mip_array[mask]
        mip_min = np.percentile(vals, 5)
        mip_max = np.percentile(vals, 99.8)
        #
        mip_array[mip_array < mip_min] = mip_min
        mip_array[mip_array > mip_max] = mip_max
        mip = (mip_array - mip_min) / (mip_max - mip_min)
        # mip = mip_array.astype(np.uint16)
        mip = (mip * (2 ** 16 - 1)).astype(np.uint16)

        img = Image.fromarray(mip)
        out_path = png_folder / f"{zarr_path.stem.replace('_z', '')}.png"
        img.save(out_path)
        print(f"Finished processing {zarr_path.name}.")
