"""Transforms module

A set of base transforms for the MANTRA dataset. We make use of such
transformations in `our paper <https://openreview.net/pdf?id=X6y5CC44HM>`__
to enable the training on different neural-network architectures.
"""

import math

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import degree


class SimplexRandomTransform(T.BaseTransform):
    """Add random features to `simplex_dim` simplices
    with a `feature_dim` dimension.

    We check the triangulation to derive the number of
    `simplex_dim` dimensional simplices.
    """

    def __init__(self, simplex_dim: int, feature_dim: int = 8):
        super().__init__()
        self.featur_dim = feature_dim
        self.k = simplex_dim

    def forward(self, data):
        assert "triangulation" in data, "Field 'triangulation` not found"

        top_simps = set([tuple(s) for s in data.triangulation])
        k_dim_simps = 0

        # For each top-simplex we count the simplices that need to exists
        # due to the closure property, we just count for each
        # NOTE: If we assume that all top-level simplices have the same
        # dimension we could just to len(top_simps) * math.comb(n,k)
        for top_simp in top_simps:
            k_dim_simps += math.comb(n=len(top_simp), k=self.k)

        # Create tensor on float32
        feat_tensor = torch.rand(
            size=(k_dim_simps, self.featur_dim), dtype=torch.float32
        )

        # Set tensor
        setattr(data, f"random_features_{self.k}", feat_tensor)

        return data


class NodeRandomTransform(T.BaseTransform):
    """
    Add random node features in `random_features`
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.dimension = dim

    def forward(self, data):
        assert "edge_index" in data, "No edge index in data"
        data.random_features = torch.rand(
            size=(int(data.edge_index.max().item() + 1), self.dimension)
        )


class NodeDegreeTransform(T.BaseTransform):
    """
    Add degrees of nodes as features in `degree`.
    """

    def forward(self, data):
        assert "edge_index" in data, "No edge index in data"
        deg = degree(data.edge_index[0], dtype=torch.float)
        data.degree = deg.view(-1, 1)
        return data
