import zarr
import napari
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from src.build_killi.build_utils import fit_sphere
from skimage.measure import regionprops
from tqdm.contrib.concurrent import process_map
from functools import partial
from tqdm import tqdm
import re

def get_saved_frames(output_dir):
    saved = []
    pattern = re.compile(r'frame_(\d+)\.png')
    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            saved.append(int(match.group(1)))
    return sorted(saved)

def run_wrapper(chunk_i, frame_array, image_full, mask_full, initial_angle, zoom, output_dir,
                        scale_vec, angle_vec, overwrite=False):

    write_frames = frame_array[chunk_i]

    # Create a napari viewer in 3D mode (ndisplay=3) to handle 3D stacks.
    viewer = napari.Viewer(ndisplay=3)

    # Add the initial image layer using the first frame
    layer0 = viewer.add_image(image_full[0, 1], name='lcp', rendering='mip', scale=scale_vec, contrast_limits=[0, 850])
    layer1 = viewer.add_image(image_full[0, 0], name='lcp', rendering='mip', colormap='cyan', scale=scale_vec,
                              contrast_limits=[135, 275], opacity=0.5)

    # Enable the built-in scale bar
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'µm'
    viewer.scale_bar.font_size = 18

    # Compute once if frame shape is constant:
    nz, ny, nx = image_full[0, 1].shape  # assume all frames share the same shape
    Z, Y, X = np.ogrid[:nz, :ny, :nx]
    Z = Z * scale_vec[0]
    Y = Y * scale_vec[1]
    X = X * scale_vec[2]

    run_write_napari = partial(
        write_napari_frames,
        viewer=viewer,
        image_full=image_full,
        mask_full=mask_full,
        layer1=layer1,
        layer0=layer0,
        initial_angle=initial_angle,
        zoom=zoom,
        output_dir=output_dir,
        scale_vec=scale_vec,
        angle_vec=angle_vec,
        X=X,
        Y=Y,
        Z=Z
    )

    saved_frames = set(get_saved_frames(output_dir))
    # frames_to_process = [frame for frame in write_frames if frame not in saved_frames]

    # else:
    # counter = 0
    lim = 10
    start_flag = True
    for frame in tqdm(write_frames, f"Writing frames from {write_frames[0]} to {write_frames[-1]}"):
        do_flag = np.any([frame not in saved_frames, overwrite])
        """
        Loop through each frame and generate the screenshots.
        """
        if do_flag:
            # Call the function to write the napari frame
            if start_flag:
                dm, sdf = run_write_napari(frame, save_flag=False)
                start_flag = False
                prev_frame = frame
            elif (frame-prev_frame) >= lim:
                dm, sdf = run_write_napari(frame)
                prev_frame = frame
            else:
                _, _ = run_write_napari(frame, dist_mask=dm, sdf=sdf)


    viewer.close()

def write_napari_frames(i, viewer, image_full, mask_full, layer1, layer0, initial_angle, output_dir,
                        scale_vec, angle_vec, zoom, X, Y, Z, save_flag=True,
                        time_text_padding=20, start_hpf=26, tres=1.5/60, dist_mask=None, sdf=None):


    # Load the current 3D frame (a 3D stack)
    frame = image_full[i]
    mask_frame = mask_full[i]

    props = regionprops(mask_frame, spacing=scale_vec)
    points = np.array([prop.centroid for prop in props])

    # fit sphere and get SH info
    # coeffs, fitted_center, fitted_radius = fit_sphere_and_sh(points, L_max=15)

    # apply fade to more distance points
    nls_frame = frame[1].copy()
    lcp_frame = frame[0].copy()

    if dist_mask is None:
        fitted_center, fitted_radius, _, _ = fit_sphere(points)
        ######


        # Assume im is your 3D volume (we only need its shape here)
        # nz, ny, nx = nls_frame.shape

        C = fitted_center
        # Create coordinate grids in a memory-efficient way using np.ogri

        # Compute the Euclidean distance from each voxel to the center C
        distance_from_center = np.sqrt((X - C[2]) ** 2 + (Y - C[1]) ** 2 + (Z - C[0]) ** 2)

        # The signed distance is the distance from the center minus the radius
        sdf = distance_from_center - fitted_radius
        sdf[sdf < 0] = 1
        # sdf_norm = np.max(sdf) - sdf
        # sdf_norm = sdf_norm / np.max(sdf_norm)
        # rand_array = sdf #+np.random.rand(sdf.shape[0], sdf.shape[1], sdf.shape[2])
        dist_mask = (1 + sdf) ** (-.5)

    f95 = np.percentile(nls_frame[mask_frame > 0], 75)  # mean of the inside sphere
    nls_norm = np.multiply(nls_frame.astype(np.float64), dist_mask)
    lcp_norm = lcp_frame.copy()
    if i < 1000:
        lcp_o = 0.01
    elif i < 1100:
        lcp_o = (i - 1000) / 100 * 0.52  # linearly increase opacity from 0.01 to 1.0 over 100 frames
        lcp_norm[sdf > 30] = 0
    else:
        lcp_o = 0.52
        lcp_norm[sdf > 30] = 0

    # viewer.add_image(nls_norm, scale=scale_vec, contrast_limits=[0, 850])
    # viewer.add_image(lcp_frame, scale=scale_vec, contrast_limits=[135, 200], colormap="green", opacity=0.6)
    # Update the layer's data (this releases the previous frame from memory)
    layer0.data = nls_norm
    layer0.contrast_limits = [0, f95]  # Update contrast limits to match the 95th percentile of the inside sphere
    layer1.data = lcp_frame
    layer1.opacity = lcp_o

    # Update the viewer's perspective by setting the camera roll.
    # This rotates the view (i.e. the perspective) about the Z axis without modifying the data.
    viewer.camera.angles = tuple([initial_angle[0], initial_angle[1] + angle_vec[i], initial_angle[2]])
    viewer.camera.zoom = zoom

    # Capture a screenshot of the current viewer canvas (excluding UI elements)
    screenshot = viewer.screenshot(canvas_only=True)

    stage = np.round(start_hpf + (i * tres), 1)
    time_text = f"{stage} hpf"
    # Draw the frame text at the top-left corner with some padding.
    # Convert the screenshot (a NumPy array) to a PIL Image for annotation.
    img = Image.fromarray(screenshot)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("arial.ttf", size=22)
    draw.text((time_text_padding, time_text_padding), time_text, font=font, fill="white")

    # Save the screenshot as a PNG file with 300 dpi resolution
    # print("Saving screenshot...")
    # if save_flag:
    output_path = os.path.join(output_dir, f'frame_{i:03d}.png')
    img.save(output_path, dpi=(300, 300))

    return dist_mask, sdf


if __name__ == "__main__":

    os.environ["QT_API"] = "pyqt5"
    overwrite = False
    # get filepaths
    root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\killi_tracker\\"
    # d_root = "D:\\Nick\\killi_tracker\\"
    project = "20250311_LCP1-NLSMSC"

    zpath = os.path.join(root, "built_data", "zarr_image_files", project + "_fused.zarr")
    mpath = os.path.join(root, "built_data", "mask_stacks", project + "_mask_fused.zarr")

    mask = zarr.open(mpath, mode="r")
    image_full = zarr.open(zpath, mode="r")

    # make output path
    output_dir = os.path.join(root, "figures", project, "screenshots_cyan", "")
    os.makedirs(output_dir, exist_ok=True)

    # set params
    initial_angle = (-3.8195498904405216, 8.89243987179023, 105.88077856425232)
    zoom = 0.58
    n_revs = 3
    frame_increment = 1
    num_frames = image_full.shape[0]
    final_angle = n_revs * 360
    angle_vec = np.linspace(0, final_angle, num_frames)

    # call parallel workers
    n_chunks = 8
    n_workers = 8
    frame_slices = np.array_split(np.arange(0, num_frames, frame_increment), n_chunks)

    scale_vec = tuple([image_full.attrs['PhysicalSizeZ'], image_full.attrs['PhysicalSizeY'], image_full.attrs['PhysicalSizeX']])

    run_run_wrapper = partial(run_wrapper,
                              frame_array=frame_slices,
                              image_full=image_full,
                              mask_full=mask,
                              initial_angle=initial_angle,
                              zoom=zoom,
                              output_dir=output_dir,
                              scale_vec=scale_vec,
                              angle_vec=angle_vec,
                              overwrite=overwrite)


    process_map(run_run_wrapper, range(n_chunks), max_workers=n_workers, chunksize=1)


    # napari.run()

    print("Check")

