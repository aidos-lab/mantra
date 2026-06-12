"""Tests for ``mantra.representations.dual_graph``.

Covers the dual-graph construction and both ``feature_propagation``
branches (none vs. an attribute mapped onto the dual nodes), which is the
path touched by the pyg>=2.7.0 feature-propagation fix.
"""

from torch_geometric.data import Data

from mantra.representations.dual_graph import DualGraph
from mantra.representations.simplicial_connectivity import (
    IncidenceSimplicialComplex,
)
from mantra.transforms.attribute_transform import NodeRandomTransform

# Boundary of a tetrahedron: 4 triangles, each adjacent to the other 3.
TETRAHEDRON_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
# Two triangles sharing the edge (1, 2).
TWO_TRIANGLES = [[1, 2, 3], [1, 2, 4]]


def _data(triangulation):
    return Data(triangulation=triangulation, dimension=2)


class TestDualGraphStructure:
    def test_tetrahedron_node_and_edge_count(self):
        out = DualGraph()(_data(TETRAHEDRON_TRI))
        # 4 triangles -> 4 dual nodes; K4 -> 6 undirected = 12 directed.
        assert out.n_vertices == 4
        assert out.edge_index.shape[1] == 12

    def test_two_triangles_share_one_edge(self):
        out = DualGraph()(_data(TWO_TRIANGLES))
        assert out.n_vertices == 2
        assert out.edge_index.shape[1] == 2  # one shared edge, both directions


class TestDualGraphFeaturePropagation:
    def test_no_features_leaves_no_x(self):
        # feature_propagation=None -> group_node_attrs=None, no ``x`` grouped.
        out = DualGraph(feature_propagation=None)(_data(TETRAHEDRON_TRI))
        assert "x" not in out

    def test_propagates_named_attribute_onto_dual_nodes(self):
        dim = 5
        data = _data(TETRAHEDRON_TRI)
        data = IncidenceSimplicialComplex(signed=False)(data)
        data = NodeRandomTransform(dim=dim, propagate=True)(data)

        out = DualGraph(feature_propagation="random_features")(data)

        # One row per dual node (top simplex), ``dim`` columns.
        assert out.x.shape == (4, dim)
