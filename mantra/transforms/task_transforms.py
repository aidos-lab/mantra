"""Task transforms module

A set of transforms for MANTRA that serve the purpouse of
specifying different targets for the possible tasks.

"""

import torch
import torch_geometric.transforms as T

from torch_geometric.data import Data

NAME_TO_CLASS_2M = {
    "Klein bottle": 0,
    "RP^2": 1,
    "T^2": 2,
    "S^2": 3,
    "": 4,
    "#^2 RP^2": 5,
    "#^3 RP^2": 6,
    "#^4 RP^2": 7,
    "#^5 RP^2": 8,
}


class OrientableToClassTransform(T.BaseTransform):
    """
    Encode the orientability target as a binary target
    with type `long`.
    """

    def forward(self, data: Data):
        data.orientable = torch.tensor(data.betti_numbers)[..., -1]
        data.y = data.orientable.long()
        return data


class NameToClass2MTransform:
    """
    Encode the homemorphism type (`name`) as a nominal target for 2-manifolds.
    """

    def __init__(self):
        self.class_dict = NAME_TO_CLASS_2M

    def forward(self, data: Data):
        assert "name" in data
        data.y = torch.tensor(self.class_dict[data.name])
        return data


class BettiToClassTransform(T.BaseTransform):
    """
    Encode the Betti number (genus) target for 2 and 3 manifolds
    as a vector with the corresponding number of elements (3, 4).
    """

    def __init__(self, manifold_dim: int = 2):
        assert (
            manifold_dim == 2 or manifold_dim == 3
        ), "Only 2 and 3 manifolds are supported"
        self.manifold_dim = manifold_dim

    def forward(self, data: Data):
        data.y = torch.tensor(data.betti_numbers, dtype=torch.float).view(
            1, self.manifold_dim + 1
        )
        return data


class BinaryHomeomorphicTransform(T.BaseTransform):
    """
    Encode as a binary label if two triangulations are homeomorphic
    to the same manifold.
    """

    def __int__(self):
        pass

    def forward(self, data: Data):
        assert "triangulation" in data, "No triangulation in this object"
        assert data.triangulation.shape[0] == 2, "Need pairwise tensors of triangulations"

        data.y = torch.tensor([data.name[0] == data.name[1]], dtype=torch.long)

        return data






