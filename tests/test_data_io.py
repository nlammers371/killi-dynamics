import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
zarr = pytest.importorskip("zarr")

if "bioio" not in sys.modules:
    bioio_stub = types.ModuleType("bioio")

    class _PlaceholderBioImage:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("BioImage must be monkeypatched in tests")

    bioio_stub.BioImage = _PlaceholderBioImage
    sys.modules["bioio"] = bioio_stub

for name in ("bioio_czi", "bioio_ome_zarr"):
    sys.modules.setdefault(name, types.ModuleType(name))

from src.data_io import czi_export


class DummyBioImage:
    def __init__(self, path: str):
        self.path = Path(path)
        # encode the time index in the data to ensure unique writes
        scalar = int(self.path.stem.split("(")[1].split(")")[0])
        self.data = np.full((3, 4, 5), scalar, dtype=np.uint16)
        self.physical_pixel_sizes = np.array([1.0, 1.0, 1.0])


def test_export_czi_to_zarr_creates_store(tmp_path, monkeypatch):
    monkeypatch.setattr(czi_export, "BioImage", DummyBioImage)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for t in ("0001", "0002"):
        (raw_dir / f"Test({t}).czi").write_bytes(b"")

    save_root = tmp_path
    czi_export.export_czi_to_zarr(
        raw_data_root=raw_dir,
        file_prefix="Test",
        project_name="proj",
        save_root=save_root,
        tres=60,
        par_flag=False,
        resampling_scale=np.array([1.0, 1.0, 1.0]),
        overwrite_flag=True,
    )

    zarr_path = save_root / "built_data" / "zarr_image_files" / "proj.zarr"
    assert zarr_path.exists()

    store = zarr.open(zarr_path.as_posix(), mode="r")
    assert store.shape == (2, 3, 4, 5)
    assert store[0].max() == 1
    assert store[1].max() == 2

