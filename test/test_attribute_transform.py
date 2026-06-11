"""Tests for ``mantra.transforms.attribute_transform``.

These cover :class:`NodeRandomTransform` (both the plain edge-index mode
and the ``propagate`` mode that distributes random features across all
simplex ranks) and :class:`NodeDegreeTransform`.
"""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge

from mantra.representations.simplicial_connectivity import (
    IncidenceSimplicialComplex,
)
from mantra.transforms.attribute_transform import (
    NodeDegreeTransform,
    NodeRandomTransform,
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


def _incidence_data():
    """A ``Data`` object carrying ``incidence_*`` matrices."""
    data = Data(triangulation=TETRAHEDRON_TRI, dimension=2)
    return IncidenceSimplicialComplex(signed=True)(data)


class TestNodeRandomTransformPlain:
    def test_default_is_not_propagate(self):
        assert NodeRandomTransform().propagate is False

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
    def test_features_per_rank(self):
        data = _incidence_data()
        result = NodeRandomTransform(dim=5, propagate=True)(data)
        rf = result.random_features
        # One entry per simplex rank present (vertices, edges, faces).
        assert set(rf.keys()) == {0, 1, 2}
        assert rf[0].shape == (4, 5)  # 4 vertices
        assert rf[1].shape == (6, 5)  # 6 edges
        assert rf[2].shape == (4, 5)  # 4 faces

    def test_is_a_plain_dict(self):
        # Regression: previously a ``defaultdict(torch.tensor)`` whose
        # factory would raise if a missing key were accessed.
        rf = NodeRandomTransform(propagate=True)(_incidence_data())
        rf = rf.random_features
        assert type(rf) is dict
        with pytest.raises(KeyError):
            _ = rf[99]

    def test_requires_incidence_matrices(self):
        data = Data(triangulation=TETRAHEDRON_TRI, dimension=2)
        with pytest.raises(AssertionError, match="No incidence matrices"):
            NodeRandomTransform(propagate=True)(data)


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
