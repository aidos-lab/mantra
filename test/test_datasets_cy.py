"""Tests for the parquet-based CY dataset."""

import torch

from mantra.datasets import CalabiYau


def _load(tmp_path, make_cy_parquet, cy_rows, **kwargs):
    return CalabiYau(
        root=str(tmp_path / "data"),
        local_path=make_cy_parquet(cy_rows),
        **kwargs,
    )


class TestCY:
    def test_roundtrip(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)

        assert len(dataset) == len(cy_rows)

        data = dataset[0]

        # Simplices are converted to the 1-indexed MANTRA convention,
        # `dimension` holds the topological dimension.
        assert data.triangulation == [
            [1, 2, 3],
            [1, 3, 4],
            [1, 4, 5],
            [1, 5, 2],
        ]
        assert int(data.dimension) == 2
        assert int(data.n_vertices) == 5

        assert data.coords.dtype == torch.float32
        assert data.coords.shape == (5, 2)

        # Extra parquet columns become attributes.
        assert int(data.h11) == 6
        assert int(data.h12) == 46
        assert int(dataset[1].h11) == 7

    def test_limit(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows, limit=1)

        assert len(dataset) == 1
        assert "limit_1" in dataset.processed_dir

        # The limited variant must not shadow the full dataset.
        full = _load(tmp_path, make_cy_parquet, cy_rows)
        assert len(full) == len(cy_rows)

    def test_name(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows, name="variant_a")

        assert "variant_a" in dataset.processed_dir
        assert len(dataset) == len(cy_rows)

        # The named variant lives in its own processed directory and
        # must not shadow the default dataset.
        full = _load(tmp_path, make_cy_parquet, cy_rows)
        assert "variant_a" not in full.processed_dir
        assert len(full) == len(cy_rows)
