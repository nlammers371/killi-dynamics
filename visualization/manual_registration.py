import napari
import numpy as np
import zarr
import scipy.ndimage as ndi
import os
from skimage.registration import phase_cross_correlation
from scipy.ndimage import rotate
from tqdm import tqdm


def shift_rotate(array, shift, angle):
    # Rotate the moving image
    rotated = rotate(array, angle, axes=(1, 2), reshape=False)

    out = ndi.shift(rotated, (shift), order=1)

    return out

def phase_cross_correlation_with_rotation(reference, moving, angle_range, angle_step):

    rotate_list = []
    shift_list = []
    corr_list = []
    angle_list = list(np.arange(angle_range[0], angle_range[1], angle_step))

    for angle in tqdm(angle_list):

        # Rotate the moving image
        rotated = rotate(moving, angle, axes=(1, 2), reshape=False)

        # Compute cross-correlation
        shift, error, _ = phase_cross_correlation(
            reference,
            rotated,
            normalization=None,
            upsample_factor=2,
            overlap_ratio=0.05,
        )
        # shift, error, _ = phase_cross_correlation(reference, rotated)
        corr = -error  # Higher correlation means better alignment

        rotate_list.append(rotated)
        corr_list.append(corr)
        shift_list.append(shift)

    return rotate_list, corr_list, shift_list, angle_list

if __name__ == '__main__':

    # load zarr image file
    zarr_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\zarr_image_files\\"
    side1_name = "20241114_LCP1-NLSMSC_side1.zarr"
    side2_name = "20241114_LCP1-NLSMSC_side2.zarr"

    # open zar files
    image_data1 = zarr.open(os.path.join(zarr_path, side1_name), mode="r")
    image_data2 = zarr.open(os.path.join(zarr_path, side2_name), mode="r")

    # get voxel size
    scale_vec = tuple([image_data1.attrs['PhysicalSizeZ'],
                       image_data1.attrs['PhysicalSizeY'],
                       image_data1.attrs['PhysicalSizeX']])

    # thresh = 11400

    # pull frame
    frame_i = 200
    channel_i = 1

    data_zyx1 = np.squeeze(image_data1[frame_i, channel_i])
    data_zyx2 = np.squeeze(image_data2[frame_i, channel_i])

    # experiment with manual alignment
    data_zyx2_a = data_zyx2[::-1, :, ::-1]

    # generate zero-padded versions of each
    z_overlap = 35
    full_shape = list(data_zyx1.shape)
    z_dim_orig1 = full_shape[0]
    z_dim_orig2 = data_zyx2.shape[0]
    full_shape[0] = data_zyx1.shape[0]*2 - z_overlap

    data_full1 = np.zeros(tuple(full_shape), dtype=np.uint16)
    data_full2 = np.zeros(tuple(full_shape), dtype=np.uint16)

    data_full2[:z_dim_orig2, :, :] = data_zyx2_a
    data_full1[-z_dim_orig1:, :, :] = data_zyx1

    # ALIGN
    z_align_size = 100
    # mask1 = np.zeros(data_full1.shape, dtype=np.bool_)
    # mask2 = np.zeros(data_full2.shape, dtype=np.bool_)
    # mask1[-z_dim_orig2:-(z_dim_orig1 - z_align_size + 1), :, :] = True
    # mask2[z_dim_orig2-z_align_size:z_dim_orig2, :, :] = True
    align1 = data_zyx1[:z_align_size, :, :]
    align2 = data_zyx2_a[-z_align_size:, :, :]

    shift, error, _ = phase_cross_correlation(
        align1,
        align2,
        normalization=None,
        upsample_factor=2,
        overlap_ratio=0.05,
    )

    # make array to store full sphere
    full_shape = list(data_zyx1.shape)
    full_shape[0] = int(np.ceil((data_zyx1.shape[0] + data_zyx1.shape[0])/10)*10)
    data_full1 = np.zeros(tuple(full_shape), dtype=np.uint16)
    data_full2 = data_full1.copy()

    # add arrays
    data_full1[-data_zyx1.shape[0]:, :, :] = data_zyx1
    data_full2[:data_zyx2.shape[0]:, :, :] = data_zyx2_a

    # apply shift
    shift_corrected = shift.copy()
    shift_corrected[0] = shift_corrected[0] + z_align_size
    data_full2_shift = ndi.shift(data_full2, (shift_corrected), order=1)


    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(data_full1, scale=scale_vec, opacity=0.7, colormap="gray", contrast_limits=[350, 1200])
    viewer.add_image(data_full2_shift, scale=scale_vec, opacity=0.7, colormap="cyan", contrast_limits=[250, 2000])
    # viewer.camera.angles = (0, 0, 0)
    viewer.window.add_plugin_dock_widget(plugin_name='napari-animation')
    viewer.layers[0].blending = 'translucent'
    viewer.layers[1].blending = 'translucent'

    from napari_animation import Animation

    # Create an Animation object linked to the viewer
    animation = Animation(viewer)
    # Create an Animation object linked to the viewer
    # Number of frames in the movie
    n_frames = 30
    # viewer.camera.angles = (0, 0, 0)  # Reset angles to a consistent starting point

    # Add the first keyframe
    animation.capture_keyframe()
    zangle_init = viewer.camera.angles[1]
    # Rotate the camera and capture frames
    for i in range(1, n_frames + 1):
        angle = 360 * (i / n_frames)  # Compute the angle
        viewer.camera.angles = (viewer.camera.angles[0], zangle_init + angle, viewer.camera.angles[2])
        animation.capture_keyframe()

    # Export the animation with interpolation
    # animation.animate("embryo_rotation.mp4", fps=30)

    # Save the animation to a file
    save_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\tracking\\20240611_NLS-Kikume_24hpf_side2\\tracking_jordao_20240918\\well0000\\track_0000_1600\\figures\\"
    animation.animate(save_path + "embryo_rotation2.mp4", fps=32)
    # viewer.add_labels(im_label, scale=scale_vec)
    #
    # # load zarr image file
    # zarr_path = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\built_data\\zarr_image_files\\"
    # side1_name = "20241114_LCP1-NLSMSC_side1.zarr"
    # side2_name = "20241114_LCP1-NLSMSC_side2.zarr"
    #
    # # open zar files
    # image_data1 = zarr.open(os.path.join(zarr_path, side1_name), mode="r")
    # image_data2 = zarr.open(os.path.join(zarr_path, side2_name), mode="r")
    #
    # # get voxel size
    # scale_vec = tuple([image_data1.attrs['PhysicalSizeZ'],
    #                    image_data1.attrs['PhysicalSizeY'],
    #                    image_data1.attrs['PhysicalSizeX']])
    #
    # thresh = 11400
    #
    # # pull frame
    # frame_i = 0
    # channel_i = 1
    #
    # data_zyx1 = np.squeeze(image_data1[frame_i, channel_i])
    # data_zyx2 = np.squeeze(image_data2[frame_i, channel_i])
    #
    # # experiment with manual alignment
    # data_zyx2_a = data_zyx2[::-1, :, ::-1]
    #
    # # generate zero-padded versions of each
    # z_overlap = 35
    # full_shape = list(data_zyx1.shape)
    # z_dim_orig1 = full_shape[0]
    # z_dim_orig2 = data_zyx2.shape[0]
    # full_shape[0] = data_zyx1.shape[0] * 2 - z_overlap
    #
    # data_full1 = np.zeros(tuple(full_shape), dtype=np.uint16)
    # data_full2 = np.zeros(tuple(full_shape), dtype=np.uint16)
    #
    # data_full2[:z_dim_orig2, :, :] = data_zyx2_a
    # data_full1[-z_dim_orig1:, :, :] = data_zyx1
    #
    # # ALIGN
    # z_align_size = 100
    # align1 = data_full1[z_dim_orig2 - z_align_size:-(z_dim_orig1 - z_align_size + 1), :, :]
    # align2 = data_full2[z_dim_orig2 - z_align_size:-(z_dim_orig1 - z_align_size + 1), :, :]
    #
    # shift, error, _ = phase_cross_correlation(
    #     align1,
    #     align2,
    #     normalization=None,
    #     upsample_factor=1,
    #     overlap_ratio=0.05,
    # )
    #
    # data_full2_shift = ndi.shift(align2, shift, order=1)
    # # angle_range = (-4, 5)
    # # rotate_list, corr_list, shift_list, angle_list = phase_cross_correlation_with_rotation(align1, align2,
    # #                                                                                        angle_range, angle_step=2)
    # # Compute cross-correlation
    # # shift, error, _ = phase_cross_correlation(
    # #     align1,
    # #     align2,
    # #     normalization=None,
    # #     upsample_factor=2,
    # #     overlap_ratio=0.05,
    # # )
    #
    # # data_full2_shift = shift_rotate(data_full2, shift=shift_list[2], angle=0)
    #
    # viewer = napari.Viewer(ndisplay=3)
    # viewer.add_image(align1, scale=scale_vec, opacity=0.7, colormap="gray", contrast_limits=[250, 2000])
    # viewer.add_image(data_full2_shift, scale=scale_vec, opacity=0.7, colormap="cyan", contrast_limits=[250, 2000])
    # # viewer.add_labels(im_label, scale=scale_vec)

