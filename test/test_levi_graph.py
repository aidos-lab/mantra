"""Tests for ``mantra.representations.levi_graph``.

Covers the bipartite structure (0-simplices vs. maximal simplices with
the documented 0-indexing), the ``feature_propagation=None`` default
(parity with the pre-propagation implementation), and coordinate-feature
propagation with exact per-node values.
"""

import torch
from torch_geometric.data import Data

from mantra.representations import LeviGraph
from mantra.transforms import CoordinateEmbedding, SelectFeatures

# Boundary of a tetrahedron with the top simplices deliberately given in
# non-lexicographic order (each tuple itself internally sorted, as in
# the CY data), pinning the index alignment of the simplex partition.
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


def _edge_set(data):
    return set(map(tuple, data.edge_index.t().tolist()))


def _coordinate_data(triangulation):
    """Attach per-rank barycentric coordinate features to a complex."""
    data = Data(
        triangulation=triangulation, dimension=2, vertices=VERTICES.clone()
    )
    data = CoordinateEmbedding(propagate=True)(data)
    return SelectFeatures(
        src="coordinate_embedding",
        dst="coordinate_embedding_{d}",
        representation="sc",
    )(data)


class TestLeviGraphStructure:
    def test_two_triangles(self):
        out = LeviGraph()(Data(triangulation=TWO_TRIANGLES, dimension=2))

        # 4 vertices + 2 top simplices.
        assert int(out.n_vertices) == 6
        # Vertex v becomes node v - 1; top simplex i becomes node 4 + i.
        assert out.simplex == [[0], [1], [2], [3], [0, 1, 2], [0, 1, 3]]
        # Exactly one (undirected) edge per vertex-in-top incidence.
        assert _edge_set(out) == {
            (0, 4), (4, 0),
            (1, 4), (4, 1),
            (2, 4), (4, 2),
            (0, 5), (5, 0),
            (1, 5), (5, 1),
            (3, 5), (5, 3),
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


class TestLeviGraphCoordinatePropagation:
    def _levi(self):
        return LeviGraph(feature_propagation="coordinate_embedding")(
            _coordinate_data(TETRAHEDRON_TRI_SCRAMBLED)
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
