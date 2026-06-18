"""Tests for ``mantra.datasets.cy``."""

import pytest
import torch

from mantra.datasets import CY


def test_local_parquet_loads_dataset(make_cy_parquet, cy_rows, tmp_path):
    path = make_cy_parquet(cy_rows)
    ds = CY(str(tmp_path / "root"), local_path=path)
    assert len(ds) == len(cy_rows)
    d = ds[0]
    # ``dimension`` is the size of a top simplex ([0, 1, 2, 3] -> 4).
    assert d.dimension == 4
    assert d.vertices.dtype == torch.float32
    assert tuple(d.vertices.shape) == (4, 3)
    assert ds.raw_file_names == ["manifolds.parquet"]
    assert ds.processed_file_names == ["data.pt"]


def test_debug_mode_loads_and_marks_root(make_cy_parquet, cy_rows, tmp_path):
    path = make_cy_parquet(cy_rows)
    ds = CY(str(tmp_path / "root"), local_path=path, debug=True)
    assert len(ds) == len(cy_rows)
    assert "cy/debug" in ds.root


def test_download_without_local_path_raises(tmp_path):
    with pytest.raises(NotADirectoryError, match="not implemented"):
        CY(str(tmp_path / "root"))


def test_pre_filter_and_pre_transform_applied(
    make_cy_parquet, cy_rows, tmp_path
):
    def pre_filter(d):
        return True

    def pre_transform(d):
        d.tagged = True
        return d

    path = make_cy_parquet(cy_rows)
    ds = CY(
        str(tmp_path / "root"),
        local_path=path,
        pre_filter=pre_filter,
        pre_transform=pre_transform,
    )
    assert len(ds) == len(cy_rows)
    assert all(d.tagged for d in ds)


def test_add_version_to_root_branches():
    obj = CY.__new__(CY)
    obj.version = "latest"
    obj.debug = False
    assert obj._add_version_to_root() == "/cy/"
    obj.debug = True
    assert obj._add_version_to_root() == "/cy/debug/"
    obj.version = "v1"
    obj.debug = False
    assert obj._add_version_to_root() == "/cy/v1/"
    obj.debug = True
    assert obj._add_version_to_root() == "/cy/v1/debug/"
