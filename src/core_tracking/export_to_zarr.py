import dask
import dask.array as da
import aicsimageio
import numpy as np
import json

def create_dummy_data(shape):
    return da.random.random(shape, chunks=(100, 100, 100))

def save_metadata(metadata, metadata_file):
    # Save metadata to a JSON file
    with open(metadata_file, 'w') as json_file:
        json.dump(metadata, json_file)

def load_metadata(metadata_file):
    # Load metadata from a JSON file
    with open(metadata_file, 'r') as json_file:
        metadata = json.load(json_file)
    return metadata

def export_dask_array_to_ome_zarr(array, ome_zarr_path, array_name, metadata_file):
    # Convert Dask array to NumPy array for export to OME-Zarr
    array_np = array.compute()

    # Save metadata to a JSON file
    save_metadata(metadata_file, metadata_file)

    # Export the array to OME-Zarr
    aicsimageio.writers.ome_zarr.write_ome_zarr(ome_zarr_path, {array_name: array_np}, metadata=metadata_file)

def import_ome_zarr_file(ome_zarr_path):
    # Read OME-Zarr file
    zarr_reader = aicsimageio.readers.ome_zarr.OmeZarrReader(ome_zarr_path)

    # Access a specific dataset (replace 'array_0' with the actual array name)
    array_0 = zarr_reader.get_dataset("array_0")

    # Convert NumPy array to Dask array
    dask_array_0 = da.from_array(array_0, chunks=array_0.shape)

    return dask_array_0

if __name__ == "__main__":
    # Create a list of 3D Dask arrays
    array_list = [create_dummy_data((100, 100, 100)) for _ in range(5)]

    # Specify the path to the output OME-Zarr file and metadata file
    ome_zarr_file_path = "output.ome.zarr"
    metadata_file_path = "metadata.json"

    # Specify time index and pixel resolution
    time_index = 0
    resolution_xyz = (0.1, 0.1, 0.5)  # Adjust according to your data

    # Export each Dask array to the same OME-Zarr file one at a time
    for i, dask_array in enumerate(array_list):
        # Create a unique array name for each array
        array_name = f"array_{i}"

        # Create metadata for each array
        metadata = {
            "SizeT": 1,  # Assuming there is only one timepoint per array, adjust if needed
            "Time": [time_index],
            "PhysicalSizeX": resolution_xyz[0],
            "PhysicalSizeY": resolution_xyz[1],
            "PhysicalSizeZ": resolution_xyz[2],
        }

        # Export the Dask array to the OME-Zarr file
        export_dask_array_to_ome_zarr(dask_array, ome_zarr_file_path, array_name, metadata)

        print(f"Array {i} exported to {ome_zarr_file_path} as {array_name}")

    # Import the OME-Zarr file and access a Dask array
    imported_dask_array = import_ome_zarr_file(ome_zarr_file_path)

    # Load metadata from the JSON file
    loaded_metadata = load_metadata(metadata_file_path)

    # Perform operations on the imported Dask array and metadata if needed
    # ...

    print("Imported Dask array shape:", imported_dask_array.shape)
    print("Loaded metadata:", loaded_metadata)