"""Tests for the parquet-based CY dataset and its companion transforms."""

import torch

from mantra.datasets import CY
from mantra.transforms import (
    CoordinateEmbedding,
    CreateRegressionLabels,
    SelectFeatures,
)


def _load(tmp_path, make_cy_parquet, cy_rows, **kwargs):
    return CY(
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
        assert data.triangulation == [[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 2]]
        assert int(data.dimension) == 2
        assert int(data.n_vertices) == 5

        assert data.vertices.dtype == torch.float32
        assert data.vertices.shape == (5, 2)

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


class TestCoordinateEmbedding:
    def test_plain(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)
        data = CoordinateEmbedding(propagate=False)(dataset[0])

        assert torch.equal(data.coordinate_embedding, data.vertices)

    def test_propagate(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)
        data = CoordinateEmbedding(propagate=True)(dataset[0])

        embedding = data.coordinate_embedding
        assert set(embedding.keys()) == {0, 1, 2}

        assert torch.equal(embedding[0], data.vertices)

        # 8 edges (4 boundary + 4 to the apex), 4 triangles; barycenters
        # live in coordinate space.
        assert embedding[1].shape == (8, 2)
        assert embedding[2].shape == (4, 2)

        # The barycenter of the lexicographically first triangle
        # (1, 2, 3) is the mean of its vertex coordinates.
        expected = data.vertices[[0, 1, 2]].mean(dim=0)
        assert torch.allclose(embedding[2][0], expected)

    def test_select_features_sc(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)
        data = CoordinateEmbedding(propagate=True)(dataset[0])
        data = SelectFeatures(
            src="coordinate_embedding", dst=None, representation="sc"
        )(data)

        for rank, count in enumerate([5, 8, 4]):
            assert data[f"x_{rank}"].shape == (count, 2)


class TestCreateRegressionLabels:
    def test_vector_target(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)
        transform = CreateRegressionLabels(sources=["h11", "h12"])

        data = transform(dataset[0])

        assert data.y.dtype == torch.float32
        assert data.y.shape == (1, 2)
        assert data.y.tolist() == [[6.0, 46.0]]

        # Interface compatibility with `CreateLabels`.
        assert transform.label_to_index == {}

    def test_scalar_source(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)
        data = CreateRegressionLabels(sources="h12")(dataset[0])

        assert data.y.shape == (1, 1)
        assert data.y.item() == 46.0

    def test_sum_sources(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)
        transform = CreateRegressionLabels(
            sources=["h11", "h12"], sum_sources=True
        )

        data = transform(dataset[1])

        assert data.y.shape == (1, 1)
        assert data.y.item() == 50.0
