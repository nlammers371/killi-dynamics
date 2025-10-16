import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
zarr = pytest.importorskip("zarr")

from src.segmentation.mask_builders import perform_li_segmentation


def test_perform_li_segmentation_writes_masks(tmp_path):
    image_store = zarr.open(
        (tmp_path / "image.zarr").as_posix(),
        mode="w",
        shape=(1, 4, 4, 4),
        dtype=np.float32,
        chunks=(1, 4, 4, 4),
    )
    image_store[0] = 1.0

    stack_store = zarr.open(
        (tmp_path / "stack.zarr").as_posix(),
        mode="w",
        shape=(1, 2, 4, 4, 4),
        dtype=np.uint16,
        chunks=(1, 1, 4, 4, 4),
    )
    stack_store.attrs["thresh_levels"] = {}

    aff_store = zarr.open(
        (tmp_path / "aff.zarr").as_posix(),
        mode="w",
        shape=(1, 4, 4, 4),
        dtype=np.uint16,
        chunks=(1, 4, 4, 4),
    )
    aff_store.attrs["thresh_levels"] = {}

    li_df = pd.DataFrame({"li_thresh": [0.5]})

    flag = perform_li_segmentation(
        time_int=0,
        li_df=li_df,
        image_zarr=image_store,
        nuclear_channel=0,
        multichannel_flag=False,
        stack_zarr=stack_store,
        aff_zarr=aff_store,
        preproc_flag=False,
        n_thresh=2,
        thresh_factors=(0.5, 1.5),
    )

    assert flag == 1
    assert aff_store[0].sum() > 0
    assert "0" in stack_store.attrs["thresh_levels"]

