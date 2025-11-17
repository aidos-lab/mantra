import torch

import torch_geometric.transforms as T
from torch_geometric.data import Data


class TriangulationToFaceTransform:
    """
    Transforms tetrahedra to faces.
    Expects a triangulation of shape [4,N] and
    returns the faces of shape [3,M].

    NOTE: It will contain duplicate triangles with different ordering.
    Hence the result is not a "minimal" triangulation. When subsequently
    converting the triangles to edges, this will not pose a problem as
    the FaceToEdge transform by default creates undirected edges.
    """

    def __init__(self, remove_triangulation: bool = False) -> None:
        self.remove_triangulation = remove_triangulation

    def __call__(self, data):
        idx = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        if hasattr(data, "triangulation"):
            assert data.triangulation is not None
            face = torch.cat(
                [torch.tensor(data.triangulation)[i] for i in idx], dim=1
            )

            # Remove duplicate triangles in
            data.face = torch.unique(face, dim=1)

            if self.remove_triangulation:
                data.triangulation = None
            else:
                data.triangulation = torch.tensor(data.triangulation)

        return data


class SetNumNodesTransform(T.BaseTransform):
    """
    Convert the `n_vertices` attribute to `num_nodes`.
    """

    def forward(self, data: Data):
        assert "n_vertices" in data
        data.num_nodes = data.n_vertices
        return data
