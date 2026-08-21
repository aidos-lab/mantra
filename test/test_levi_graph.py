"""Tests for ``mantra.representations.graph.levi_graph``.

Covers the bipartite structure (0-simplices vs. maximal simplices with
the documented 0-indexing), the ``feature_propagation=None`` default
(parity with the pre-propagation implementation), and feature
propagation with exact per-node values.
"""

import pytest
import torch
from torch_geometric.data import Data

from mantra.representations import LeviGraph
from mantra.transforms import PropagateConvexComb

# Boundary of a tetrahedron with the top simplices deliberately given in
# non-lexicographic order (each tuple itself internally sorted), pinning
# the index alignment of the simplex partition.
TETRAHEDRON_TRI_SCRAMBLED = [[2, 3, 4], [1, 2, 4], [1, 2, 3], [1, 3, 4]]
# The same top simplices, 0-indexed and lexicographically sorted: the
# order the Levi graph must use for the maximal-simplex partition.
TETRAHEDRON_TOPS_SORTED = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
# Two triangles sharing the edge (1, 2).
TWO_TRIANGLES = [[1, 2, 3], [1, 2, 4]]

# Vertex coordinates whose x-coordinates are powers of two: barycenters
# of *distinct* vertex subsets are pairwise distinct, so a misindexed
# feature lookup cannot produce the correct value by accident.
VERTICES = torch.tensor([[1.0, 0.1], [2.0, 0.2], [4.0, 0.4], [8.0, 0.8]])


@pytest.fixture
def transform():
    return LeviGraph()


@pytest.fixture
def single_triangle():
    return [[1, 2, 3]]


@pytest.fixture
def two_triangles():
    return [[1, 2, 3], [1, 2, 4]]


def _edge_set(data):
    return set(map(tuple, data.edge_index.t().tolist()))


def _propagated_data(triangulation):
    """Attach per-rank barycentric features `x_0`, `x_1`, `x_2`."""
    data = Data(
        triangulation=triangulation, dimension=2, coords=VERTICES.clone()
    )
    return PropagateConvexComb(source="coords")(data)


class TestLeviGraph:
    def _make_data(self, triangulation):
        data = Data(triangulation=triangulation)
        return data

    def _cnt_nodes(self, data):
        nodes_seen = set()

        # TODO: Ugly can't come up with a smarter way
        for top_simp in data.triangulation:
            nodes_seen.update(top_simp)
        return len(nodes_seen)

    @pytest.mark.parametrize("triangles", ["single_triangle", "two_triangles"])
    def test_node_count(self, transform, triangles, request):
        triangles = request.getfixturevalue(triangles)

        data = transform(self._make_data(triangles))

        assert "triangulation" in data

        cnt_nodes = self._cnt_nodes(data)

        assert data.n_vertices == cnt_nodes + len(data.triangulation)


class TestLeviGraphStructure:
    def test_two_triangles(self):
        out = LeviGraph()(Data(triangulation=TWO_TRIANGLES, dimension=2))

        # 4 vertices + 2 top simplices.
        assert int(out.n_vertices) == 6
        # Vertex v becomes node v - 1; top simplex i becomes node 4 + i.
        assert out.simplex == [[0], [1], [2], [3], [0, 1, 2], [0, 1, 3]]
        # Exactly one (undirected) edge per vertex-in-top incidence.
        assert _edge_set(out) == {
            (0, 4),
            (4, 0),
            (1, 4),
            (4, 1),
            (2, 4),
            (4, 2),
            (0, 5),
            (5, 0),
            (1, 5),
            (5, 1),
            (3, 5),
            (5, 3),
        }

    def test_tetrahedron_bipartite_edges(self):
        out = LeviGraph()(
            Data(triangulation=TETRAHEDRON_TRI_SCRAMBLED, dimension=2)
        )

        assert int(out.n_vertices) == 8

        # Top-simplex nodes follow the lexicographic order of the top
        # simplices, not the scrambled input order.
        assert out.simplex[4:] == TETRAHEDRON_TOPS_SORTED

        # Exact edge-set equality: every top simplex connects to
        # precisely its own vertices, and no vertex-vertex or top-top
        # edges exist (the graph is bipartite).
        expected = set()
        for i, top in enumerate(TETRAHEDRON_TOPS_SORTED):
            for v in top:
                expected.add((v, 4 + i))
                expected.add((4 + i, v))
        assert _edge_set(out) == expected


class TestLeviGraphDefault:
    def test_no_propagation_adds_no_features(self):
        # ``feature_propagation`` defaults to ``None`` and must then
        # reproduce the pre-propagation behavior: the bipartite graph
        # plus the per-node ``simplex`` attribute, but no ``x``.
        out = LeviGraph()(
            Data(triangulation=TETRAHEDRON_TRI_SCRAMBLED, dimension=2)
        )

        assert "x" not in out
        assert out.edge_index.shape == (2, 24)
        assert int(out.n_vertices) == 8
        assert out.simplex == [[0], [1], [2], [3]] + TETRAHEDRON_TOPS_SORTED

    def test_no_propagation_ignores_present_features(self):
        # Propagated features stay untouched unless asked for.
        out = LeviGraph()(_propagated_data(TETRAHEDRON_TRI_SCRAMBLED))

        assert "x" not in out
        assert out.x_0.shape == (4, 2)


class TestLeviGraphFeaturePropagation:
    def _levi(self):
        return LeviGraph(feature_propagation="x")(
            _propagated_data(TETRAHEDRON_TRI_SCRAMBLED)
        )

    def test_vertex_nodes_carry_vertex_coordinates(self):
        out = self._levi()

        assert out.x.shape == (8, 2)
        # Node i < 4 is vertex i + 1 and carries its raw coordinates.
        assert torch.equal(out.x[:4], VERTICES)

    def test_top_simplex_nodes_carry_their_barycenter(self):
        out = self._levi()

        assert out.simplex[4:] == TETRAHEDRON_TOPS_SORTED
        for row, top in zip(out.x[4:], out.simplex[4:]):
            expected = VERTICES[top].mean(dim=0)
            assert torch.allclose(row, expected)

        # All 8 features are pairwise distinct, so index misalignment
        # between the two partitions cannot pass the checks above.
        assert out.x.unique(dim=0).shape[0] == 8

    def test_missing_rank_tensor_raises(self):
        data = Data(
            triangulation=TWO_TRIANGLES, dimension=2, x_0=VERTICES.clone()
        )
        # Rank-2 features are required for the maximal simplices.
        with pytest.raises(AttributeError):
            LeviGraph(feature_propagation="x")(data)
