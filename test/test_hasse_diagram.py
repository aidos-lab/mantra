"""Tests for ``mantra.representations.hasse_diagram``.

The main purpose here is a regression guard: importing and instantiating
:class:`HasseDiagram` used to raise ``NameError: name 'Union' is not
defined`` because the type hint referenced ``Union`` while only
``Optional`` was imported. Function annotations are evaluated at class
definition time, so the failure happened on import.
"""

from torch_geometric.data import Data

from mantra.representations.hasse_diagram import HasseDiagram
from mantra.representations.simplicial_connectivity import (
    IncidenceSimplicialComplex,
)
from mantra.transforms.attribute_transform import NodeRandomTransform

# Boundary of a tetrahedron: 4 vertices, 6 edges, 4 faces -> 14 nodes.
TETRAHEDRON_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]


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
    dim = 5
    data = Data(triangulation=TETRAHEDRON_TRI, dimension=2)
    data = IncidenceSimplicialComplex(signed=False)(data)
    data = NodeRandomTransform(dim=dim, propagate=True)(data)

    out = HasseDiagram(feature_propagation="random_features")(data)

    # ``from_networkx`` groups the named node attribute into ``x``:
    # one row per Hasse node (14), ``dim`` columns.
    assert out.x.shape == (14, dim)
