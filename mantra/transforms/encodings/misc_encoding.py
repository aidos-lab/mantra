"""Transforms module

A set of base transforms for the MANTRA dataset. We make use of such
transformations in `our paper <https://openreview.net/pdf?id=X6y5CC44HM>`__
to enable the training on different neural-network architectures.
"""

from itertools import combinations

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
        self.feature_dim = feature_dim
        self.k = simplex_dim

    def forward(self, data):
        assert "triangulation" in data, "Field 'triangulation` not found"

        top_simps = set([tuple(s) for s in data.triangulation])
        k_dim_simps = set()

        # For each top-simplex we count the simplices that need to exists
        # due to the closure property, we just count for each
        for top_simp in top_simps:
            assert self.k <= len(
                top_simp
            ), f"There's simplex_dim={self.k} exceeds the size of a triangulation"
            # Here we do self.k + 1 since we selected simplex dimension k, which
            # mean simplices composed of k+1  elements
            k_dim_simps.update(s for s in combinations(top_simp, r=self.k + 1))

        # Create tensor on float32
        feat_tensor = torch.rand(
            size=(len(list(k_dim_simps)), self.feature_dim),
            dtype=torch.float32,
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
        return data


class NodeDegreeTransform(T.BaseTransform):
    """
    Add degrees of nodes as features in `degree`.
    """

    def forward(self, data):
        assert "edge_index" in data, "No edge index in data"
        deg = degree(data.edge_index[0], dtype=torch.float)
        data.degree = deg.view(-1, 1)
        return data


class ScalarFeatures(T.BaseTransform):
    """Collect scalar attributes into a single feature vector.

    This transform assembles per-sample scalar attributes (e.g.
    `n_vertices` and `genus` of a triangulation, or any other
    per-sample count) into a `scalar_features` tensor
    of shape `(1, k)`. It provides a graph-level input for baseline
    models that do not consume the triangulation itself; assign it to
    `x` with `SelectFeatures(src="scalar_features")`.
    """

    def __init__(self, sources):
        """Create new scalar feature transform.

        Parameters
        ----------
        sources : str or list of str
            Scalar attributes to collect, in order. Each attribute must
            be present in the data, either as a Python scalar (the
            `pre_transform` path) or as a one-element tensor (the
            `transform` path after collation).
        """
        super().__init__()

        self.sources = [sources] if isinstance(sources, str) else list(sources)

    def forward(self, data):
        """Assign scalar feature vector for a given `data` object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object. All source attributes must be present.

        Returns
        -------
        torch_geometric.data.Data
            Data object with a new `scalar_features` key of shape
            `(1, k)` and dtype `float32`.
        """
        values = []
        for source in self.sources:
            assert (
                source in data
            ), f"Source attribute '{source}' is not present in data"

            value = data[source]
            if isinstance(value, torch.Tensor):
                value = value.item()
            values.append(float(value))

        data.scalar_features = torch.tensor(values, dtype=torch.float32).view(
            1, -1
        )
        return data
