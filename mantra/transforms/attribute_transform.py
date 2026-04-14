"""Transforms module

A set of base transforms for the MANTRA dataset. We make use of such
transformations in `our paper <https://openreview.net/pdf?id=X6y5CC44HM>`__
to enable the training on different neural-network architectures.
"""

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import degree

from collections import defaultdict


class NodeRandomTransform(T.BaseTransform):
    """
    Add random node features in `random_features`
    """

    def __init__(self, dim: int = 8, propagate: bool = False):
        super().__init__()
        self.dimension = dim
        self.propagate = propagate

    def forward(self, data):
        if not self.propagate:
            assert "edge_index" in data, "No edge index in data"
            data.random_features = torch.rand(
                size=(int(data.edge_index.max().item() + 1), self.dimension)
            )
        else: # Propagate random features to all simplices
            # All incidence matrices required
            incidence_list = [k for k in data.keys() if "incidence" in k]

            assert len(incidence_list) > 0, "No incidence matrices found in data"

            # Sort by rank `r`
            sorted(incidence_list, key = lambda x: int(x.split('_')[1]))

            random_features = defaultdict(torch.tensor)

            for inc_m in incidence_list:
                I = getattr(data, inc_m)
                r_to = int(inc_m.split('_')[1])
                r_from = r_to - 1
                # Case for incidence_0
                if r_from < 0:
                    continue

                random_features[r_from] = torch.rand(
                    size=(I.shape[0], self.dimension)
                )

                random_features[r_to] = torch.rand(
                    size=(I.shape[1], self.dimension)
                )
            data.random_features = random_features
        return data


class SimplexRandomTransform(T.BaseTransform):
    """
    Add `random_features` to simplices based on incidence matrices
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.dimension = dim

    def forward(self, data):


class NodeDegreeTransform(T.BaseTransform):
    """
    Add degrees of nodes as features in `degree`.
    """

    def forward(self, data):
        assert "edge_index" in data, "No edge index in data"
        deg = degree(data.edge_index[0], dtype=torch.float)
        data.degree = deg.view(-1, 1)
        return data
