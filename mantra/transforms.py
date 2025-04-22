"""Transforms module

A set of base transforms for the MANTRA dataset. We make use of such
transformations in `our paper <https://openreview.net/pdf?id=X6y5CC44HM>`__
to enable the training on different neural-network architectures.
"""

import torch

from torch_geometric.transforms import Compose
from torch_geometric.transforms import FaceToEdge
from torch_geometric.transforms import OneHotDegree

from torch_geometric.utils import degree


class NodeIndex:
    """
    This transform ensures the compatibility with `pytorch-geometric` by
    changing the node/vertex indices to be zero-indexed.
    """

    def __call__(self, data):
        data.face = torch.tensor(data.triangulation).T - 1
        return data


class RandomNodeFeatures:
    """
    Adds random node features to the dataset. The main purpose behind
    this transformation is to ensure compatibility with architectures
    that require node features, while also showing their respective
    shortcomings. In our dataset, unlike many others, node coordinates
    and the triangulations themselves are fully decoupled.
    """

    def __init__(self, dimension=8):
        self.dimension = dimension

    def __call__(self, data):
        data.x = torch.rand(size=(data.face.max() + 1, self.dimension))
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
