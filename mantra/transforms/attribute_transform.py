"""Attribute transforms module

A set of attribute transforms for the MANTRA dataset. We make use of such
transformations in `our paper <https://openreview.net/pdf?id=X6y5CC44HM>`__
to enable the training on different neural-network architectures.
"""

import torch

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torch_geometric.transforms import FaceToEdge
from torch_geometric.transforms import OneHotDegree

from torch_geometric.utils import degree


class NodeRandomTransform(T.BaseTransform):
    """
    Add random node features
    """

    def __init__(self, dim: int = 8):
        self.dimension = dim

    def forward(self, data):
        assert "edge_index" in data, "No edge index in data"
        data.x = torch.rand(
            size=(int(data.edge_index.max().item() + 1), self.dimension)
        )
        return data


class NodeDegreeTransform(T.BaseTransform):
    """
    Add degrees of nodes as features in `x`.
    """

    def forward(self, data):
        assert "edge_index" in data, "No edge index in data"
        deg = degree(data.edge_index[0], dtype=torch.float)
        data.x = deg.view(-1, 1)
        return data
