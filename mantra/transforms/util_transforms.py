from itertools import combinations

import torch
from torch_geometric.transforms import BaseTransform


class PropagateConvexComb(BaseTransform):
    """Propagates the features of a tensor `source` describing
    0-simplices to higher-simplices based on the barycenter
    of the feature.
    """

    def __init__(self, source: str = "x"):
        """
        Parameters
        ----------
        source : str
            Name of the source `Tensor` to be contained in each `Data`
            obj. Defaults to `x`
        """
        self.source = source

    def forward(self, data):

        assert (
            "triangulation" in data
        ), "Data object is missing `triangulation`"
        assert (
            self.source in data
        ), f"Data object is missing source tensor `{self.source}`"

        x = getattr(data, self.source)
        triangulation = getattr(data, "triangulation")

        X = torch.as_tensor(x, dtype=torch.float32)

        simplices = set([tuple(s) for s in triangulation])
        max_dim = len(next(iter(simplices)))

        for simplex in triangulation:
            for dim in range(1, max_dim):
                simplices.update(s for s in combinations(simplex, r=dim))

        # To sort lexicographically, we need to turn this back into
        # something mutable.
        simplices = list(simplices)
        simplices.sort()
        simplices.sort(key=len)

        # Dictionary containing the new attribute keys
        values = {"x_0": X}

        for dim in range(1, max_dim):
            simplices_ = [s for s in simplices if len(s) == dim + 1]

            # Every simplex of this rank has `dim + 1` vertices, so the
            # index tensor is rectangular; subtract one to map the
            # 1-indexed vertex labels to tensor rows.
            idx = torch.tensor(simplices_) - 1

            # Calculate all barycenters of the current rank at once.
            # TODO: Change this to another type of combination function
            values[f"x_{dim}"] = X[idx].mean(dim=1)

        for k, v in values.items():
            data[k] = v
        return data
