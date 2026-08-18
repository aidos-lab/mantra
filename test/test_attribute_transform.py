"""Tests for ``mantra.transforms.attribute_transform``.

These cover :class:`NodeRandomTransform` (plain edge-index mode) and :class:`SimplexRandomTransform`
that assigns random features to fixed simplices
and :class:`NodeDegreeTransform`.
"""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge

from mantra.representations import IncidenceSimplicialComplex
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


class TestNodeRandomTransformPropagate:
    """``propagate=True`` derives per-rank features from the incidences."""

    def _incidence_data(self):
        data = Data(triangulation=TETRAHEDRON_TRI, dimension=2)
        return IncidenceSimplicialComplex(signed=False)(data)

    def test_one_tensor_per_rank_with_incidence_shapes(self):
        data = self._incidence_data()
        result = NodeRandomTransform(dim=7, propagate=True)(data)
        features = result.random_features

        # One tensor per rank, keyed by ascending rank.
        assert isinstance(features, dict)
        assert list(features.keys()) == [0, 1, 2]

        # ``incidence_k`` maps rank-k simplices (columns) to their
        # rank-(k-1) faces (rows): rank k gets ``incidence_k.shape[1]``
        # rows, rank 0 gets ``incidence_1.shape[0]`` rows, i.e. one row
        # per node.
        assert features[0].shape == (4, 7)
        assert features[0].shape[0] == result.incidence_1.shape[0]
        assert features[1].shape == (6, 7)
        assert features[1].shape[0] == result.incidence_1.shape[1]
        assert features[2].shape == (4, 7)
        assert features[2].shape[0] == result.incidence_2.shape[1]

        assert all(v.dtype == torch.float32 for v in features.values())

    def test_incidence_0_is_skipped(self):
        # ``incidence_0`` has no rank below it and is skipped: a data
        # object carrying only this matrix satisfies the incidence
        # assertion but yields an empty feature dictionary.
        data = self._incidence_data()
        only_zero = Data(incidence_0=data.incidence_0)
        result = NodeRandomTransform(dim=3, propagate=True)(only_zero)
        assert result.random_features == {}

    def test_requires_incidence_matrices(self):
        data = Data(triangulation=TETRAHEDRON_TRI, dimension=2)
        with pytest.raises(AssertionError, match="No incidence matrices"):
            NodeRandomTransform(propagate=True)(data)


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
