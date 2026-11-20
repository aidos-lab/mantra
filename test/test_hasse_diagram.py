"""Tests for ``mantra.representations.hasse_diagram``.

The main purpose here is a regression guard: importing and instantiating
:class:`HasseDiagram` used to raise ``NameError: name 'Union' is not
defined`` because the type hint referenced ``Union`` while only
``Optional`` was imported. Function annotations are evaluated at class
definition time, so the failure happened on import.
"""

from itertools import combinations

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_networkx

from mantra.representations import HasseDiagram
from mantra.transforms import (
    PropagateConvexComb,
    SimplexRandomTransform,
)

# Boundary of a tetrahedron: 4 vertices, 6 edges, 4 faces -> 14 nodes.
TETRAHEDRON_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
# The same complex with the top simplices deliberately given in
# non-lexicographic order (each tuple itself internally sorted, as in
# the CY data), pinning the alignment between node order and the
# lexicographically ordered per-rank feature tensors.
TETRAHEDRON_TRI_SCRAMBLED = [[2, 3, 4], [1, 2, 4], [1, 2, 3], [1, 3, 4]]
# One 4-simplex: 5 vertices, 10 edges, 10 triangles, 5 tetrahedra, 1 cell.
FOUR_SIMPLEX_TRI = [[1, 2, 3, 4, 5]]

# Vertex coordinates whose x-coordinates are powers of two: barycenters
# of *distinct* vertex subsets are pairwise distinct, so a misindexed
# feature lookup cannot produce the correct value by accident.
VERTICES = torch.tensor([[1.0, 0.1], [2.0, 0.2], [4.0, 0.4], [8.0, 0.8]])


def _coordinate_data(triangulation):
    """Attach per-rank barycentric coordinate features to a complex."""
    data = Data(
        triangulation=triangulation, dimension=2, coords=VERTICES.clone()
    )
    return PropagateConvexComb(source="coords")(data)


def test_instantiation_does_not_raise():
    # Regression for the ``Union`` NameError on the type annotation.
    transform = HasseDiagram(feature_propagation=None)
    assert transform.feature_propagation is None


def test_forward_builds_full_hasse_diagram():
    data = Data(triangulation=TETRAHEDRON_TRI, dimension=2)
    out = HasseDiagram(feature_propagation=None)(data)
    # One node per simplex of every rank: 4 + 6 + 4 = 14.
    assert int(out.n_vertices) == 14
    assert out.edge_index.shape[0] == 2


def test_forward_propagates_per_rank_features_onto_nodes():
    # End-to-end: per-rank random features (the feature this PR adds) are
    # mapped onto every node of the Hasse diagram via ``feature_propagation``.
    feature_dim = 5
    data = Data(triangulation=TETRAHEDRON_TRI, dimension=2)
    random_all_simp_trf = Compose(
        [
            SimplexRandomTransform(simplex_dim=i, feature_dim=feature_dim)
            for i in range(len(TETRAHEDRON_TRI[0]))
        ]
    )
    data = random_all_simp_trf(data)
    out = HasseDiagram(feature_propagation="random_features")(data)

    # ``from_networkx`` groups the named node attribute into ``x``:
    # one row per Hasse node (14), ``dim`` columns.
    assert out.x.shape == (14, feature_dim)


def test_forward_propagates_coordinate_barycenters_per_node():
    # Value-level check with deterministic per-rank features: every
    # Hasse node must carry the feature of the simplex it represents.
    data = _coordinate_data(TETRAHEDRON_TRI_SCRAMBLED)
    out = HasseDiagram(feature_propagation="x")(data)

    assert out.x.shape == (14, 2)
    # ``simplex`` holds the (0-indexed) vertices of each node, aligned
    # with the rows of ``x``; all ranks are present.
    assert len(out.simplex) == 14
    assert sorted(len(s) for s in out.simplex) == [1] * 4 + [2] * 6 + [3] * 4

    for i, simplex in enumerate(out.simplex):
        # Barycenter of the node's own vertices; for rank-0 nodes this
        # is the raw vertex coordinate.
        expected = VERTICES[simplex].mean(dim=0)
        assert torch.allclose(out.x[i], expected)

    # All 14 features are pairwise distinct (powers-of-two coordinates),
    # so a misindexed lookup cannot pass the loop above.
    assert out.x.unique(dim=0).shape[0] == 14


@pytest.mark.parametrize(
    "triangulation, max_rank, n_nodes",
    [
        (TETRAHEDRON_TRI, 0, 4),
        (TETRAHEDRON_TRI, 1, 10),
        (TETRAHEDRON_TRI, 2, 14),
        (TETRAHEDRON_TRI, 4, 14),
        (TETRAHEDRON_TRI, None, 14),
        (FOUR_SIMPLEX_TRI, 1, 15),
        (FOUR_SIMPLEX_TRI, 2, 25),
        (FOUR_SIMPLEX_TRI, 3, 30),
        (FOUR_SIMPLEX_TRI, 4, 31),
        (FOUR_SIMPLEX_TRI, None, 31),
    ],
)
def test_max_rank_caps_the_node_count(triangulation, max_rank, n_nodes):
    out = HasseDiagram(max_rank=max_rank)(Data(triangulation=triangulation))

    assert int(out.n_vertices) == n_nodes
    assert len(out.simplex) == n_nodes
    cap = 4 if max_rank is None else max_rank
    assert max(len(s) for s in out.simplex) - 1 <= cap


def test_negative_max_rank_raises():
    with pytest.raises(ValueError, match="max_rank"):
        HasseDiagram(max_rank=-1)


@pytest.mark.parametrize("max_rank", [1, 2, 3, 4, None])
@pytest.mark.parametrize(
    "triangulation", [TETRAHEDRON_TRI_SCRAMBLED, FOUR_SIMPLEX_TRI]
)
def test_vertices_first_then_rank_by_rank(triangulation, max_rank):
    out = HasseDiagram(max_rank=max_rank)(Data(triangulation=triangulation))

    n = max(max(s) for s in triangulation)
    assert out.simplex[:n] == [[v] for v in range(n)]
    assert out.vertex_mask.dtype == torch.bool
    assert out.vertex_mask.tolist() == [True] * n + [False] * (
        len(out.simplex) - n
    )
    assert out.simplex == sorted(out.simplex, key=lambda s: (len(s), s))


@pytest.mark.parametrize("max_rank", [1, 2, 3, 4, None])
def test_edges_are_the_covering_relations(max_rank):
    out = HasseDiagram(max_rank=max_rank)(Data(triangulation=FOUR_SIMPLEX_TRI))

    edges = {(i, j) for i, j in out.edge_index.t().tolist() if i < j}
    expected = {
        (i, j)
        for i, low in enumerate(out.simplex)
        for j, high in enumerate(out.simplex)
        if len(high) == len(low) + 1 and set(low) < set(high)
    }
    assert edges == expected
    assert out.edge_index.shape == (2, 2 * len(expected))


def test_max_rank_one_is_the_vertex_edge_incidence_graph():
    out = HasseDiagram(max_rank=1)(Data(triangulation=TETRAHEDRON_TRI))

    edges = [s for s in out.simplex if len(s) == 2]
    expected = {(v, 4 + i) for i, edge in enumerate(edges) for v in edge}
    assert {
        (i, j) for i, j in out.edge_index.t().tolist() if i < j
    } == expected


@pytest.mark.parametrize(
    "triangulation", [TETRAHEDRON_TRI_SCRAMBLED, FOUR_SIMPLEX_TRI]
)
def test_uncapped_diagram_is_the_full_face_poset(triangulation):
    out = HasseDiagram()(Data(triangulation=triangulation))

    faces = {
        face
        for top in triangulation
        for k in range(1, len(top) + 1)
        for face in combinations(top, k)
    }
    full = nx.Graph()
    full.add_nodes_from(faces)
    full.add_edges_from(
        (facet, face)
        for face in faces
        for facet in combinations(face, len(face) - 1)
        if facet
    )
    graph = to_networkx(
        Data(edge_index=out.edge_index, num_nodes=len(out.simplex)),
        to_undirected=True,
    )
    assert nx.is_isomorphic(graph, full)


@pytest.mark.parametrize("max_rank", [1, 2, None])
def test_max_rank_keeps_the_features_of_each_simplex(max_rank):
    data = _coordinate_data(TETRAHEDRON_TRI_SCRAMBLED)
    out = HasseDiagram(feature_propagation="x", max_rank=max_rank)(data)

    assert out.x.shape == (len(out.simplex), 2)
    assert torch.equal(out.x[out.vertex_mask], VERTICES)
    for i, simplex in enumerate(out.simplex):
        assert torch.allclose(out.x[i], VERTICES[simplex].mean(dim=0))


@pytest.mark.parametrize(
    "max_rank, width", [(1, 2), (2, 3), (4, 5), (None, 3)]
)
def test_rank_feature_appends_a_one_hot_of_the_rank(max_rank, width):
    data = _coordinate_data(TETRAHEDRON_TRI_SCRAMBLED)
    out = HasseDiagram(
        feature_propagation="x", max_rank=max_rank, rank_feature=True
    )(data)

    assert out.x.shape == (len(out.simplex), 2 + width)
    assert out.x.dtype == VERTICES.dtype
    ranks = torch.tensor([len(s) - 1 for s in out.simplex])
    assert torch.equal(
        out.x[:, 2:], torch.nn.functional.one_hot(ranks, width).float()
    )
    for i, simplex in enumerate(out.simplex):
        assert torch.allclose(out.x[i, :2], VERTICES[simplex].mean(dim=0))


def test_rank_feature_without_propagation_adds_no_features():
    out = HasseDiagram(rank_feature=True)(Data(triangulation=TETRAHEDRON_TRI))

    assert "x" not in out
