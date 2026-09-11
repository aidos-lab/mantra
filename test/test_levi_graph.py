import pytest
import torch
from torch_geometric.data import Data

from mantra.representations import LeviGraph


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

    def test_node_types_mark_the_bipartition(self, transform, two_triangles):
        data = transform(self._make_data(two_triangles))

        # Vertices first (four of them), then the two triangles.
        assert torch.equal(
            data.node_type, torch.tensor([0, 0, 0, 0, 1, 1])
        )
        # Every edge joins a vertex to a triangle.
        src, dst = data.edge_index
        assert (data.node_type[src] != data.node_type[dst]).all()
