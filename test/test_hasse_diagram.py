"""Tests for ``mantra.representations.hasse_diagram``.

The main purpose here is a regression guard: importing and instantiating
:class:`HasseDiagram` used to raise ``NameError: name 'Union' is not
defined`` because the type hint referenced ``Union`` while only
``Optional`` was imported. Function annotations are evaluated at class
definition time, so the failure happened on import.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

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

# Vertex coordinates whose x-coordinates are powers of two: barycenters
# of *distinct* vertex subsets are pairwise distinct, so a misindexed
# feature lookup cannot produce the correct value by accident.
VERTICES = torch.tensor([[1.0, 0.1], [2.0, 0.2], [4.0, 0.4], [8.0, 0.8]])


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
