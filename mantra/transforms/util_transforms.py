from torch_geometric.transforms import BaseTransform
import torch
import numpy as np
from collections import defaultdict
from itertools import combinations


class PropagateConvexComb(BaseTransform):
    """ Propagates the features of a tensor `source` describing
        0-simplices to higher-simplices based on the barycenter
        of the feature.
    """
    def __init__(self, source: str  = 'x'):
        """
        Parameters
        ----------
        source : str
            Name of the source `Tensor` to be contained in each `Data`
            obj. Defaults to `x`
        """
        self.source = source

    def forward(self, data):

        assert "triangulation" in data, "Data object is missing `triangulation`"
        assert self.source in data, f"Data object is missing source tensor `{self.source}`"

        x = getattr(data, self.source)
        triangulation = getattr(data, 'triangulation')

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
        values = {
            "x_0": x
        }

        for dim in range(1, max_dim):
            simplices_ = [s for s in simplices if len(s) == dim + 1]
            M = []

            for s in simplices_:
                # View as an array to correct for the index shift; our
                # triangulation is not zero-indexed.
                s = np.asarray(s)

                # Calculate barycenter for the current simplex (i.e., one
                # row of the result matrix).
                # TODO: Change this to another type of combination function
                M.append(np.mean(x[s - 1, :], axis=0))

            M = np.asarray(M)
            values[f"x_{dim}"] = torch.from_numpy(M).to(torch.float32)

        for k, v in values.items():
            data[k] = v
        return data 