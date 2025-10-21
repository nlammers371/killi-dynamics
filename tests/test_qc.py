import pytest

np = pytest.importorskip("numpy")
zarr = pytest.importorskip("zarr")

from src.qc.mask_qc import compute_qc_keep_labels, persist_keep_labels


def test_compute_qc_keep_labels_filters_small_objects():
    mask = np.zeros((4, 4, 4), dtype=np.int32)
    mask[0, :2, :2] = 1  # small volume
    mask[2:, 2:, 2:] = 2  # larger volume
    scale_vec = (1.0, 1.0, 1.0)

    keep = compute_qc_keep_labels(
        mask,
        scale_vec,
        min_nucleus_vol=5.0,
        z_prox_thresh=-10.0,
        max_eccentricity=10.0,
        min_overlap=0.0,
    )

    assert set(keep.tolist()) == {2}


def test_persist_keep_labels(tmp_path):
    store_path = (tmp_path / "mask.zarr").as_posix()
    store = zarr.open(store_path, mode="w", shape=(1, 2, 2, 2), dtype=np.uint16, chunks=(1, 2, 2, 2))
    store.attrs["mask_keep_ids"] = {}

    persist_keep_labels(store_path, 0, [5, 7])

    reopened = zarr.open(store_path, mode="r")
    assert reopened.attrs["mask_keep_ids"]["0"] == [5, 7]
