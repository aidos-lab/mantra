import torch
from torch_geometric.data import Data

from mantra.representations import (
    AdjacencySimplicialComplex,
    IncidenceSimplicialComplex,
)
from mantra.transforms import PropagateConvexComb


class TestTensors:

    def setup_method(self):

        self.triangulation = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
        ]

        self.data = Data(
            triangulation=self.triangulation, dimension=torch.tensor([2])
        )

        self.transform_incidence = IncidenceSimplicialComplex(signed=False)
        self.transform_adjacency = AdjacencySimplicialComplex(signed=False)

    def test_incidence(self):

        self.incidence_0 = torch.zeros((5, 5))

        self.incidence_1 = torch.as_tensor(
            [  # (0,1), (0,2),(1, 2), (1,3), (2,3), (2,4), (3,4)
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ],
            dtype=torch.long,
        )

        self.incidence_2 = torch.as_tensor(
            [  # (0, 1, 2), (1, 2, 3), (2,3,4)
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.long,
        )

        data = self.transform_incidence(self.data)
        assert (
            self.incidence_0 == data.incidence_0.to_dense()
        ).all(), "Error at `incidence_0` "

        assert (
            self.incidence_1 == data.incidence_1.to_dense()
        ).all(), "Error at `incidence_1` "

        assert (
            self.incidence_2 == data.incidence_2.to_dense()
        ).all(), "Error at `incidence_2` "

    def test_adjacencies(self):
        self.adjacency_0 = torch.as_tensor(
            [
                [0.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )

        self.adjacency_1 = torch.as_tensor(
            [  # (0,1), (0,2),(1, 2), (1,3), (2,3), (2,4), (3,4)
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ],
            dtype=torch.long,
        )

        self.adjacency_2 = torch.as_tensor(
            [  # (0, 1, 2), (1, 2, 3), (2,3,4)
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.long,
        )

        data = self.transform_adjacency(self.data)
        assert (
            self.adjacency_0 == data.adjacency_0.to_dense()
        ).all(), "Error at `adjacency_0` "

        assert (
            self.adjacency_1 == data.adjacency_1.to_dense()
        ).all(), "Error at `adjacency_1` "

        assert (
            self.adjacency_2 == data.adjacency_2.to_dense()
        ).all(), "Error at `adjacency_2` "


def test_rows_follow_lexicographic_order_for_large_labels(large_label_tri):
    data = Data(triangulation=large_label_tri, dimension=2)
    data.x = torch.arange(1, 11, dtype=torch.float32).unsqueeze(1)

    data = IncidenceSimplicialComplex(signed=False)(data)
    data = PropagateConvexComb(source="x")(data)

    incidence_1 = data.incidence_1.to_dense()
    incidence_2 = data.incidence_2.to_dense()
    edges = incidence_1.T @ data.x_0 / 2
    triangles = (incidence_1 @ incidence_2 > 0).float().T @ data.x_0 / 3

    torch.testing.assert_close(data.x_1, edges)
    torch.testing.assert_close(data.x_2, triangles)
