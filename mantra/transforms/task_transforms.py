"""Task transforms module

A set of transforms for MANTRA that serve the purpouse of
specifying different targets for the possible tasks.

"""

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

# Homeomorphism-type classes for closed 2-manifolds.
#
# Layout:
#   0..8   orientable genus 0..8       (S^2, T^2, #^2 T^2, ..., #^8 T^2)
#   9..25  non-orientable genus 1..17  (RP^2, Klein bottle, #^3 RP^2, ...)
#
# "Klein bottle" and "#^2 RP^2" denote the same homeomorphism type and
# share class 10. The non-orientable side enumerates k=3..17 even though
# the current dataset release skips k in {9, 11, 13, 14}; including them
# means a future regeneration that fills those genera won't reintroduce
# a KeyError.
NAME_TO_CLASS_2M = {
    # orientable: genus 0..8
    "S^2": 0,
    "T^2": 1,
    "#^2 T^2": 2,
    "#^3 T^2": 3,
    "#^4 T^2": 4,
    "#^5 T^2": 5,
    "#^6 T^2": 6,
    "#^7 T^2": 7,
    "#^8 T^2": 8,
    # non-orientable: genus 1..17
    "RP^2": 9,
    "Klein bottle": 10,
    "#^2 RP^2": 10,  # same homeomorphism type as the Klein bottle
    "#^3 RP^2": 11,
    "#^4 RP^2": 12,
    "#^5 RP^2": 13,
    "#^6 RP^2": 14,
    "#^7 RP^2": 15,
    "#^8 RP^2": 16,
    "#^9 RP^2": 17,
    "#^10 RP^2": 18,
    "#^11 RP^2": 19,
    "#^12 RP^2": 20,
    "#^13 RP^2": 21,
    "#^14 RP^2": 22,
    "#^15 RP^2": 23,
    "#^16 RP^2": 24,
    "#^17 RP^2": 25,
}

# Homeomorphism-type classes for closed 3-manifolds.
#
# Unlike closed surfaces, closed 3-manifolds are not classified by their
# Betti numbers, so the homeomorphism type is given directly by ``name``.
# The released MANTRA 3-manifold datasets contain exactly the nine named
# types below (the members of ``manifold_types.Manifold3Type``); the keys
# here mirror that enum and the indices are dense ``0..8``.
NAME_TO_CLASS_3M = {
    "S^3": 0,
    "S^2 x S^1": 1,
    "S^2 twist S^1": 2,
    "RP^3": 3,
    "L(3,1)": 4,
    "L(4,1)": 5,
    "T^3": 6,
    "S^3/Q": 7,
    "(S^2 x S^1)#(S^2 x S^1)": 8,
}


class OrientableToClassTransform(T.BaseTransform):
    """
    Encode the orientability target as a binary target
    with type `long`.
    """

    def forward(self, data: Data):
        data.orientable = torch.tensor(data.betti_numbers)[..., -1]
        data.y = data.orientable.long().view(1)
        return data


class NameToClass2MTransform(T.BaseTransform):
    """
    Encode the homeomorphism type (`name`) as a nominal target for 2-manifolds.
    """

    def __init__(self):
        self.class_dict = NAME_TO_CLASS_2M

    def forward(self, data: Data):
        assert "name" in data
        data.y = torch.tensor([self.class_dict[data.name]], dtype=torch.long)
        return data


class NameToClass3MTransform(T.BaseTransform):
    """
    Encode the homeomorphism type (`name`) as a nominal target for
    3-manifolds.
    """

    def __init__(self):
        self.class_dict = NAME_TO_CLASS_3M

    def forward(self, data: Data):
        assert "name" in data
        data.y = torch.tensor([self.class_dict[data.name]], dtype=torch.long)
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


def canonical_dict_for(dimension: int) -> dict:
    """Return the canonical ``name -> class`` dict for ``dimension`` (2 or 3)."""
    if dimension == 2:
        return NAME_TO_CLASS_2M
    if dimension == 3:
        return NAME_TO_CLASS_3M
    raise ValueError(f"dimension must be 2 or 3, got {dimension}")


def make_label_transform(task: str, dimension: int) -> T.BaseTransform:
    """Build the canonical label transform for a classification ``task``.

    ``"name"`` dispatches to :class:`NameToClass2MTransform` /
    :class:`NameToClass3MTransform` by ``dimension``; ``"orientability"`` to
    :class:`OrientableToClassTransform`. Both assign ``data.y`` in canonical
    class space.
    """
    if task == "name":
        if dimension == 2:
            return NameToClass2MTransform()
        if dimension == 3:
            return NameToClass3MTransform()
        raise ValueError(
            f"task='name' requires dimension in (2, 3), got {dimension}"
        )
    if task == "orientability":
        return OrientableToClassTransform()
    raise ValueError(
        f"Unknown task {task!r} for canonical labelling "
        "(expected 'name' or 'orientability')."
    )


class BinaryHomeomorphicTransform(T.BaseTransform):
    """
    Encode as a binary label if two triangulations are homeomorphic
    to the same manifold.
    """

    def __int__(self):
        pass

    def forward(self, data: Data):
        assert "triangulation" in data, "No triangulation in this object"
        assert (
            data.triangulation.shape[0] == 2
        ), "Need pairwise tensors of triangulations"

        data.y = torch.tensor([data.name[0] == data.name[1]], dtype=torch.long)

        return data
