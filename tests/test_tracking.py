import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from src.tracking.workflow import reindex_mask


def test_reindex_mask_applies_lookup():
    mask = np.array(
        [
            [[0, 1], [2, 0]],
            [[0, 2], [1, 0]],
        ],
        dtype=np.uint16,
    )
    lookup = np.arange(3, dtype=np.uint16)
    lookup[1] = 5
    lookup[2] = 6

    reindex_mask(0, mask, lookup)

    assert mask[0, 0, 1] == 5
    assert mask[0, 1, 0] == 6


def test_reindex_mask_skips_when_labels_present():
    mask = np.array(
        [
            [[0, 1], [2, 0]],
        ],
        dtype=np.uint16,
    )
    lookup = np.arange(3, dtype=np.uint16)
    lookup[1] = 5
    lookup[2] = 6

    df = pd.DataFrame({"t": [0, 0], "track_id": [1, 2]})
    reindex_mask(0, mask, lookup, track_df=df)

    assert set(np.unique(mask)) == {0, 1, 2}
