import pytest
import torch
from torch_geometric.data import Data

from mantra.representations import LeviGraph
from mantra.transforms import PropagateConvexComb


@pytest.fixture
def transform():
    return LeviGraph()


@pytest.fixture
def single_triangle():
    return [[1, 2, 3]]


@pytest.fixture
def two_triangles():
    return [[1, 2, 3], [1, 2, 4]]


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
        assert data.vertex_mask.tolist() == [True] * cnt_nodes + [False] * len(
            data.triangulation
        )


# Vertex coordinates whose x-coordinates are powers of two: barycenters
# of distinct vertex subsets are pairwise distinct, so a misindexed
# feature lookup cannot produce the correct value by accident.
VERTICES = torch.tensor([[1.0, 0.1], [2.0, 0.2], [4.0, 0.4], [8.0, 0.8]])
TETRAHEDRON_TRI_SCRAMBLED = [[2, 3, 4], [1, 2, 4], [1, 2, 3], [1, 3, 4]]


def test_forward_propagates_coordinate_barycenters_per_node():
    data = Data(
        triangulation=TETRAHEDRON_TRI_SCRAMBLED,
        dimension=2,
        coords=VERTICES.clone(),
    )
    data = PropagateConvexComb(source="coords")(data)
    out = LeviGraph(feature_propagation="x")(data)

    # 4 vertex nodes first, then the 4 facets; x_1 (edges) is unused.
    assert out.x.shape == (8, 2)
    assert torch.equal(out.x[out.vertex_mask], VERTICES)
    for i, simplex in enumerate(out.simplex):
        assert torch.allclose(out.x[i], VERTICES[simplex].mean(dim=0))
    assert out.edge_index.shape == (2, 2 * 12)
