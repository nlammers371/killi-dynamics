import numpy as np
import json
import os
import zarr
import napari
import SimpleITK as sitk
from skimage.exposure import match_histograms

def fuse_multiview_experiment(image_root, side1_name, side2_name, par_flag=True, overwrite_flag=False, channel_to_use=0, n_workers=6):


    # open the two views
    zarr0_path = os.path.join(image_root, "built_data", "zarr_image_files", side1_name + '.zarr')
    zarr1_path = os.path.join(image_root, "built_data", "zarr_image_files", side2_name + '.zarr')

    zarr0 = zarr.open(zarr0_path, mode="r")
    zarr1 = zarr.open(zarr1_path, mode="r")

    # match intensity levels
    time_ind = 100
    zyx0 = sitk.GetImageFromArray(zarr0[time_ind][::-1, :, :][3:], sitk.sitkFloat32)
    zyx1 = sitk.GetImageFromArray(zarr1[time_ind], sitk.sitkFloat32)


    initial_transform = sitk.CenteredTransformInitializer(zyx0, zyx1, sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # Set up the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMeanSquares()

    # Interpolator settings.
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                                                 minStep=1e-4,
                                                                 numberOfIterations=200,
                                                                 gradientMagnitudeTolerance=1e-8)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Initial transform
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Execute the registration.
    final_transform = registration_method.Execute(sitk.Cast(zyx0, sitk.sitkFloat32),
                                                  sitk.Cast(zyx1, sitk.sitkFloat32))

    print("Check")



if __name__ == "__main__":

    # da_chunksize = (1, 207, 256, 256)
    resampling_scale = np.asarray([1.5, 1.5, 1.5])
    tres = 123.11  # time resolution in seconds

    # set path parameters
    # raw_data_root = "D:\\Syd\\231016_EXP40_LCP1_UVB_300mJ\\PreUVB_Timelapse_Raw\\"
    raw_data_root = "D:\\Syd\\240611_EXP50_NLS-Kikume_24hpf_2sided_NuclearTracking\\" #"D:\\Syd\\240219_LCP1_67hpf_to_"
    # Specify the path to the output OME-Zarr file and metadata file
    image_root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    side1_name = "20240611_NLS-Kikume_24hpf_side1"
    side2_name = "20240611_NLS-Kikume_24hpf_side2"
    overwrite = False

    fuse_multiview_experiment(image_root, side1_name, side2_name, channel_to_use=0, overwrite_flag=overwrite)