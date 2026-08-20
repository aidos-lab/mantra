"""Tests for ``mantra.representations.dual_graph``.

Covers the dual-graph construction and both ``feature_propagation``
branches (none vs. an attribute mapped onto the dual nodes), which is the
path touched by the pyg>=2.7.0 feature-propagation fix.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

from mantra.representations import DualGraph
from mantra.transforms import (
    PropagateConvexComb,
    SimplexRandomTransform,
)

# Boundary of a tetrahedron: 4 triangles, each adjacent to the other 3.
TETRAHEDRON_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
# The same complex with the top simplices deliberately given in
# non-lexicographic order (each tuple itself internally sorted, as in
# the CY data): dict-insertion order then differs from the sorted order
# the per-rank feature tensors use, so permutation bugs become visible.
TETRAHEDRON_TRI_SCRAMBLED = [[2, 3, 4], [1, 2, 4], [1, 2, 3], [1, 3, 4]]
# Two triangles sharing the edge (1, 2).
TWO_TRIANGLES = [[1, 2, 3], [1, 2, 4]]

# Vertex coordinates whose x-coordinates are powers of two: barycenters
# of *distinct* vertex subsets are pairwise distinct, so a misindexed
# feature lookup cannot produce the correct value by accident.
VERTICES = torch.tensor([[1.0, 0.1], [2.0, 0.2], [4.0, 0.4], [8.0, 0.8]])


def _data(triangulation):
    return Data(triangulation=triangulation, dimension=2)


def _coordinate_data(triangulation):
    """Attach per-rank barycentric coordinate features to a complex."""
    data = Data(
        triangulation=triangulation,
        dimension=2,
        coords=VERTICES.numpy().copy(),
    )
    data = PropagateConvexComb(source="coords")(data)
    # ``PropagateConvexComb`` returns rank 0 as the raw source array;
    # normalise so all per-rank feature tensors are torch tensors.
    data.x_0 = torch.as_tensor(data.x_0, dtype=torch.float32)
    return data


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
        feature_dim = 5
        data = _data(TETRAHEDRON_TRI)
        random_all_simp_trf = Compose(
            [
                SimplexRandomTransform(simplex_dim=i, feature_dim=feature_dim)
                for i in range(len(TETRAHEDRON_TRI[0]))
            ]
        )
        data = random_all_simp_trf(data)

        out = DualGraph(feature_propagation="random_features")(data)

        # One row per dual node (top simplex), ``dim`` columns.
        assert out.x.shape == (4, feature_dim)


class TestDualGraphCoordinateValues:
    """Exact feature values under barycentric coordinate propagation."""

    def _dual(self):
        return DualGraph(feature_propagation="x")(
            _coordinate_data(TETRAHEDRON_TRI_SCRAMBLED)
        )

    def test_node_features_are_top_simplex_barycenters(self):
        out = self._dual()

        # Dual nodes follow the lexicographic order of the top
        # simplices, not the scrambled input order.
        assert out.simplex.tolist() == [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ]

        for i in range(4):
            expected = VERTICES[out.simplex[i]].mean(dim=0)
            assert torch.allclose(out.x[i], expected)

        # All barycenters differ, so the check above is discriminating.
        assert out.x.unique(dim=0).shape[0] == 4

    def test_edge_features_are_shared_face_barycenters(self):
        out = self._dual()

        # K4 -> 12 directed edges, one feature row per edge.
        assert out.edge_attr.shape == (12, 2)

        # Walking the sorted top simplices inserts the shared faces as
        # (1,2), (1,3), (2,3), (1,4), ... into the coface dictionary:
        # *not* lexicographic order. An implementation that indexed the
        # per-rank feature tensor by dictionary order instead of sorted
        # order would therefore permute the rows checked below.
        for col, attr in zip(out.edge_index.t().tolist(), out.edge_attr):
            a, b = col
            shared = sorted(
                set(out.simplex[a].tolist()) & set(out.simplex[b].tolist())
            )
            # Adjacent top simplices share exactly one edge (2 vertices).
            assert len(shared) == 2
            expected = VERTICES[shared].mean(dim=0)
            assert torch.allclose(attr, expected)
