"""Tests for ``mantra.representations.one_skeleton``."""

from torch_geometric.data import Data

from mantra.representations.one_skeleton import OneSkeleton


def _edge_set(data):
    return set(map(tuple, data.edge_index.t().tolist()))


class TestOneSkeleton:
    def test_basic_triangle(self):
        # A single triangle (1,2,3) -> nodes 0,1,2 and all three edges.
        data = OneSkeleton()(Data(triangulation=[[1, 2, 3]]))
        assert data.num_nodes == 3
        assert _edge_set(data) == {
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 0),
            (1, 2),
            (2, 1),
        }

    def test_node_labels_are_not_permuted(self):
        # Regression: ``from_networkx`` maps nodes to consecutive integers in
        # insertion order. Adding nodes lazily while iterating over edges can
        # introduce a vertex out of order and permute the labels, breaking the
        # correspondence with the original (zero-indexed) vertex order that
        # e.g. the moment-curve embedding relies on.
        #
        # The path 1-3-2 (edges (1,3) and (2,3), no edge (1,2)) triggers the
        # lazy-insertion permutation: vertex 2 would otherwise be inserted
        # third and mapped to index 1 instead of 2.
        data = OneSkeleton()(Data(triangulation=[[1, 3], [2, 3]]))
        assert data.num_nodes == 3
        # node i must correspond to original vertex (i + 1): edges (0,2)/(1,2).
        assert _edge_set(data) == {(0, 2), (2, 0), (1, 2), (2, 1)}

    def test_triangulation_key_is_preserved(self):
        # The transform only adds keys; existing keys must remain untouched.
        data = OneSkeleton()(Data(triangulation=[[1, 2, 3]]))
        assert data.triangulation == [[1, 2, 3]]
