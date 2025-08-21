import zarr
import numpy as np
import os
import napari
from tqdm import tqdm

if __name__ == '__main__':
    # load zarr image file
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    project_name = "20241114_LCP1-NLSMSC_side1" #, "20241114_LCP1-NLSMSC_side2"]

    # get image file
    zarr_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\zarr_image_files\\"
    # zarr_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\cellpose_output\\LCP-Multiset-v1\\20240611_NLS-Kikume_24hpf_side2\\20240611_NLS-Kikume_24hpf_side2_probs.zarr"
    image_data = zarr.open(os.path.join(zarr_path, project_name + ".zarr"), mode="r")

    # get voxel size
    scale_vec = tuple([image_data.attrs['PhysicalSizeZ'],
                       image_data.attrs['PhysicalSizeY'],
                       image_data.attrs['PhysicalSizeX']])

    # get mask file
    mask_directory = os.path.join(root, "built_data", "mask_stacks", '')
    multi_mask_zarr_path = os.path.join(mask_directory, project_name + "_mask_stacks.zarr")
    mm_zarr = zarr.open(multi_mask_zarr_path, mode="r")

    start_i = 200
    stop_i = 201
    channel_i = 1
    lvl = 0

    # get mask stats
    df_list = []
    for t in tqdm(range(start_i, stop_i)):
        nz_ids = mm_zarr[t, lvl] > 0
        id_vec = mm_zarr[t, lvl][nz_ids]
        f_vec = image_data[t, channel_i][nz_ids]
        df_temp = pd.DataFrame([t] * len(id_vec), columns=["t"])
        df_temp["track_id"] = id_vec
        df_temp["fluo"] = f_vec
        df_list.append(df_temp)

    nucleus_df = pd.concat(df_list, axis=0, ignore_index=True)
    nucleus_df["volume"] = 1

    stats_df = nucleus_df.groupby(["t", "track_id"]).sum().reset_index()
    stats_df["fluo_mean"] = np.divide(stats_df["fluo"], stats_df["volume"])

    # set output path for tracking results
    experiment_date = "20240611_NLS-Kikume_24hpf_side2"
    config_name = "tracking_jordao_20240918.txt"
    model = "LCP-Multiset-v1"
    tracking_folder = config_name.replace(".txt", "")
    tracking_folder = tracking_folder.replace(".toml", "")

    well_num = 0
    start_i = 0
    stop_i = 1600

    suffix = ""
    project_path = os.path.join(root, "tracking", experiment_date, tracking_folder, f"well{well_num:04}" + suffix, "")
    project_sub_path = os.path.join(project_path, f"track_{start_i:04}" + f"_{stop_i:04}" + suffix, "")

    stats_df.to_csv(project_sub_path + "evl_stats.csv", index=False)
    # create mask to separate the different layers
    ft0 = stats_df["volume"] >= 260
    ft1 = stats_df["fluo_mean"] >= 1100
    ft2 = stats_df["volume"] < 50

    evl_ft = ft0 | ft1
    evl_ids = stats_df.loc[evl_ft, "track_id"]
    hair_ids = stats_df.loc[ft2, "track_id"]

    # clean_masks = mm_zarr.copy()
    # clean_masks[np.isin(clean_masks, hair_ids)] = 0

    evl_mask = np.zeros_like(mm_zarr[t, lvl])
    deep_mask = np.zeros_like(mm_zarr[t, lvl])

    deep_mask[mm_zarr[t, lvl] > 0] = 1
    evl_mask[np.isin(mm_zarr[t, lvl], evl_ids)] = 2
    deep_mask[np.isin(mm_zarr[t, lvl], evl_ids)] = 2
    evl_mask[np.isin(mm_zarr[t, lvl], hair_ids)] = 0
    deep_mask[np.isin(mm_zarr[t, lvl], hair_ids)] = 0

    # visualize
    import skimage as ski
    import SimpleITK as sitk
    from skimage import filters
    from skimage import morphology
    channel_i = 1
    start_i = 500
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(image_data[start_i, channel_i], scale=scale_vec)


    gauss_sigma = (1.33, 4, 4)
    image = image_data[start_i, channel_i]
    # denoised = ndi.median_filter(data_zyx, size=3)
    gaussian_background = ski.filters.gaussian(image, sigma=gauss_sigma, preserve_range=True)
    data_bkg = image - gaussian_background
    data_bkg_d = np.divide(image, gaussian_background)

    data_log = sitk.GetArrayFromImage(
        sitk.LaplacianRecursiveGaussian(sitk.GetImageFromArray(data_bkg_d), sigma=1))
    data_log_i = ski.util.invert(data_log)
    viewer.add_image(data_bkg, scale=scale_vec, name="sub background")
    viewer.add_image(data_log_i, scale=scale_vec, name="LoG")
    viewer.add_image(data_bkg_d, scale=scale_vec, name="d")
    viewer.add_labels(mm_zarr[start_i, 1], scale=scale_vec, name="LoG")

    from napari_animation import Animation
    from tqdm import tqdm
    animation = Animation(viewer)

    # Loop through each frame
    start_contrast = np.asarray([-4, 50])
    offset = np.linspace(0, 15, image_data.shape[0])
    for t, frame in enumerate(tqdm(np.arange(0, 1600, 5))):
        # print(f"Processing frame {t + 1}/{n_frames}")
        im = image_data[frame]
        # Load the current frame into Napari
        if len(viewer.layers) > 0:
            viewer.layers[0].data = im
            viewer.layers[0].contrast_limits = start_contrast-offset[frame]
        else:
            viewer.add_image(im, name="Data", contrast_limits=start_contrast-offset[frame])

        # Capture keyframe for this time point
        animation.capture_keyframe()

    # Save the animation as a video
    animation.animate("timelapse.mp4", fps=14)
    viewer.close()
    # viewer.add_image(image_data[900], scale=scale_vec, contrast_limits=start_contrast-offset[frame])
