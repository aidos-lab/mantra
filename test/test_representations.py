"""Tests for representation transforms (one_skeleton, dual_graph, hasse_diagram)."""

import pytest
import torch
from torch_geometric.data import Data

from mantra.representations.one_skeleton import OneSkeleton
from mantra.representations.dual_graph import DualGraph
from mantra.representations.hasse_diagram import HasseDiagram


# --- Fixtures ---

# Boundary of tetrahedron: 4 vertices, 4 triangles (S^2)
TETRAHEDRON_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]

# Two triangles sharing edge (1,2)
TWO_TRIANGLES = [[1, 2, 3], [1, 2, 4]]

# Boundary of 4-simplex: 5 vertices, 5 tetrahedra (S^3)
FOUR_SIMPLEX_TRI = [
    [1, 2, 3, 4],
    [1, 2, 3, 5],
    [1, 2, 4, 5],
    [1, 3, 4, 5],
    [2, 3, 4, 5],
]


def _make_data(triangulation, dimension=2, n_vertices=None):
    if n_vertices is None:
        n_vertices = len(
            set(v for s in triangulation for v in s)
        )
    return Data(
        triangulation=triangulation,
        dimension=dimension,
        n_vertices=n_vertices,
    )


class TestOneSkeleton:
    def test_tetrahedron_edge_count(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = OneSkeleton()
        result = transform(data)
        # K4 has 6 edges -> 12 directed edges
        assert result.edge_index.shape[1] == 12

    def test_tetrahedron_node_count(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = OneSkeleton()
        result = transform(data)
        assert result.edge_index.max().item() == 3  # 0-indexed: 0,1,2,3

    def test_two_triangles(self):
        data = _make_data(TWO_TRIANGLES)
        transform = OneSkeleton()
        result = transform(data)
        # 4 vertices, 5 edges -> 10 directed edges
        assert result.edge_index.shape[1] == 10

    def test_3d_triangulation(self):
        data = _make_data(FOUR_SIMPLEX_TRI, dimension=3, n_vertices=5)
        transform = OneSkeleton()
        result = transform(data)
        # K5 has 10 edges -> 20 directed edges
        assert result.edge_index.shape[1] == 20


class TestDualGraph:
    def test_tetrahedron_dual_node_count(self):
        """Boundary of tetrahedron: 4 triangles -> 4 dual nodes."""
        data = _make_data(TETRAHEDRON_TRI)
        transform = DualGraph()
        result = transform(data)
        # 4 dual vertices
        assert result.n_vertices == 4

    def test_tetrahedron_dual_edge_count(self):
        """Each pair of triangles shares an edge -> K4 dual = 6 edges."""
        data = _make_data(TETRAHEDRON_TRI)
        transform = DualGraph()
        result = transform(data)
        assert result.edge_index.shape[1] == 12  # 6 undirected = 12 directed

    def test_two_triangles_dual(self):
        """Two triangles sharing one edge -> 2 dual nodes, 1 edge."""
        data = _make_data(TWO_TRIANGLES)
        transform = DualGraph()
        result = transform(data)
        assert result.n_vertices == 2
        assert result.edge_index.shape[1] == 2


class TestHasseDiagram:
    def test_two_triangles_node_count(self):
        """Two triangles: 4 verts + 5 edges + 2 triangles = 11 nodes."""
        data = _make_data(TWO_TRIANGLES)
        transform = HasseDiagram(feature_propagation=None)
        result = transform(data)
        assert result.n_vertices == 11

    def test_tetrahedron_node_count(self):
        """Boundary of tet: 4 verts + 6 edges + 4 triangles = 14 nodes."""
        data = _make_data(TETRAHEDRON_TRI)
        transform = HasseDiagram(feature_propagation=None)
        result = transform(data)
        assert result.n_vertices == 14
