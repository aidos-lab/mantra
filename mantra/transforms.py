"""
A set of base transforms for the MANTRA dataset. In the
paper these transforms are used and are provided as base
transforms for training your own neural networks.
"""

import torch
from torch_geometric.transforms import FaceToEdge, OneHotDegree
from torch_geometric.utils import degree
from torchvision.transforms import Compose


class NodeIndex:
    def __call__(self, data):
        """
        In the base dataset, the vertex start index is 1 and is provided as a
        list. The transform converts the list to a tensor and changes the start
        index to 0, for compatibility with torch-geometric.
        """
        data.face = torch.tensor(data.triangulation).T - 1
        return data


class RandomNodeFeatures:
    def __call__(self, data):
        """
        We create an 8-dimensional vector with random numbers for each vertex.
        Often the coordinates of the graph or triangulation are tightly coupled
        with the structure of the graph, an assumtion we hope to tackle.
        """
        data.x = torch.rand(size=(data.face.max() + 1, 8))
        return data


class DegreeTransform:
    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        data.x = deg.view(-1, 1)
        return data


class DegreeTransformOneHot:
    def __init__(self):
        self.transform = Compose(
            [
                NodeIndex(),
                FaceToEdge(remove_faces=False),
                OneHotDegree(max_degree=9, cat=False),
            ]
        )
