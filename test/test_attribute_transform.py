"""Tests for ``mantra.transforms.attribute_transform``.

These cover :class:`NodeRandomTransform` (plain edge-index mode) and :class:`SimplexRandomTransform`
that assigns random features to fixed simplices
and :class:`NodeDegreeTransform`.
"""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge

from mantra.transforms.attribute_transform import (
    NodeDegreeTransform,
    NodeRandomTransform,
    SimplexRandomTransform,
)

# Boundary of a tetrahedron: a triangulated 2-sphere with
# 4 vertices, 6 edges and 4 triangular faces.
TETRAHEDRON_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]


def _edge_data():
    """A ``Data`` object carrying an ``edge_index`` (graph view)."""
    data = Data(triangulation=TETRAHEDRON_TRI, dimension=2)
    # 0-index the triangulation into faces, then derive edges.
    data.face = (torch.tensor(TETRAHEDRON_TRI).t().contiguous() - 1).long()
    data.num_nodes = 4
    return FaceToEdge(remove_faces=False)(data)


class TestNodeRandomTransformPlain:
    def test_creates_random_features(self):
        data = _edge_data()
        result = NodeRandomTransform(dim=5)(data)
        # One row per node, ``dim`` columns.
        assert result.random_features.shape == (4, 5)
        assert result.random_features.dtype == torch.float

    def test_requires_edge_index(self):
        data = Data(triangulation=TETRAHEDRON_TRI, dimension=2)
        with pytest.raises(AssertionError, match="No edge index"):
            NodeRandomTransform()(data)


class TestSimplexRandomTransform:
    def test_features_exist_1(self):
        data = _edge_data()
        transform = SimplexRandomTransform(simplex_dim=1, feature_dim=5)
        result = transform(data)

        assert "random_features_1" in result

    def test_features_exists_2(self):
        data = _edge_data()
        transform_1 = SimplexRandomTransform(simplex_dim=1, feature_dim=5)
        transform_2 = SimplexRandomTransform(simplex_dim=2, feature_dim=5)
        result = transform_2(transform_1(data))

        assert "random_features_1" in result
        assert "random_features_2" in result

    def test_features_size(self):
        data = _edge_data()
        transform_1 = SimplexRandomTransform(simplex_dim=1, feature_dim=4)
        transform_2 = SimplexRandomTransform(simplex_dim=2, feature_dim=5)
        result = transform_2(transform_1(data))

        assert result.random_features_1.shape == (6, 4)  # 6 edges
        assert result.random_features_2.shape == (4, 5)  # 4 faces


class TestNodeDegreeTransform:
    def test_creates_degree(self):
        data = _edge_data()
        result = NodeDegreeTransform()(data)
        assert result.degree.shape == (4, 1)
        # Every vertex of a tetrahedron boundary has degree 3.
        assert torch.equal(result.degree, torch.full((4, 1), 3.0))

    def test_requires_edge_index(self):
        data = Data(triangulation=TETRAHEDRON_TRI, dimension=2)
        with pytest.raises(AssertionError, match="No edge index"):
            NodeDegreeTransform()(data)
