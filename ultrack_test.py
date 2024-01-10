import napari
import numpy as np
import dask.array as da
# from napari.utils.notebook_display import nbscreenshot
from rich.pretty import pprint

# from ultrack.config import MainConfig, load_config
# from ultrack import track, to_tracks_layer, tracks_to_zarr
# from ultrack.imgproc.intensity import robust_invert
# from ultrack.imgproc.segmentation import detect_foreground
# from ultrack.utils.array import array_apply, create_zarr

from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


image_path = "http://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001_tail.ome.zarr/"

# read the image data
store = parse_url(image_path, mode="r").store
reader = Reader(parse_url(image_path))

nodes = list(reader())

# first node will be the image pixel data
image_node = nodes[0]
image_data = image_node.data

viewer = napari.view_image(image_data[2])
# viewer.add_image(image_data[2], gamma=0.7, contrast_limits=(0, 500))
# viewer.window.resize(1800, 1000)
# nbscreenshot(viewer)
napari.run()