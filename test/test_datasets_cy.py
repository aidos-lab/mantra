"""Tests for the parquet-based CY dataset and its companion transforms."""

import torch

from mantra.datasets import CalabiYau
from mantra.transforms import (
    CoordinateEmbedding,
    PropagateConvexComb,
    SelectFeatures,
)


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


class TestCoordinateEmbedding:
    def test_plain(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)
        data = CoordinateEmbedding()(dataset[0])

        assert torch.equal(data.coordinate_embedding, data.coords)

    def test_propagate(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)
        data = CoordinateEmbedding()(dataset[0])
        data = PropagateConvexComb(source="coordinate_embedding")(data)

        # Rank-0 features are the coordinates themselves.
        assert torch.equal(data.x_0, data.coords)

        # 8 edges (4 boundary + 4 to the apex), 4 triangles; barycenters
        # live in coordinate space.
        assert data.x_1.shape == (8, 2)
        assert data.x_2.shape == (4, 2)
        assert "x_3" not in data

        # The barycenter of the lexicographically first triangle
        # (1, 2, 3) is the mean of its vertex coordinates.
        expected = data.coords[[0, 1, 2]].mean(dim=0)
        assert torch.allclose(data.x_2[0], expected)

    def test_missing_coords_raises(self):
        import pytest
        from torch_geometric.data import Data

        with pytest.raises(AssertionError, match="coords"):
            CoordinateEmbedding()(Data(triangulation=[[1, 2, 3]]))

    def test_accepts_non_tensor_inputs(self):
        import numpy as np
        from torch_geometric.data import Data

        data = Data(
            coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
            h11=6,
        )
        data = CoordinateEmbedding(append_attributes=["h11"])(data)

        embedding = data.coordinate_embedding
        assert embedding.dtype == torch.float32
        assert embedding.shape == (2, 3)
        assert torch.all(embedding[:, 2] == 6.0)

    def test_select_features_graph(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)
        data = CoordinateEmbedding()(dataset[0])
        data = SelectFeatures(
            src="coordinate_embedding", dst=None, representation="graph"
        )(data)

        assert torch.equal(data.x, data.coords)


class TestCoordinateEmbeddingAppend:
    def test_append_plain(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)
        data = CoordinateEmbedding(append_attributes=["h11", "h12"])(
            dataset[0]
        )

        embedding = data.coordinate_embedding
        assert embedding.shape == (5, 4)
        assert torch.equal(embedding[:, :2], data.coords)
        assert torch.all(embedding[:, 2] == 6.0)
        assert torch.all(embedding[:, 3] == 46.0)

    def test_append_propagates_to_all_ranks(
        self, tmp_path, make_cy_parquet, cy_rows
    ):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)
        data = CoordinateEmbedding(append_attributes=["h11"])(dataset[0])
        data = PropagateConvexComb(source="coordinate_embedding")(data)

        for rank, count in enumerate([5, 8, 4]):
            values = data[f"x_{rank}"]
            assert values.shape == (count, 3)
            # Barycenters of a constant column stay constant.
            assert torch.all(values[:, 2] == 6.0)
