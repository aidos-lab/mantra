"""Task transforms module

A set of transforms for MANTRA that serve the purpouse of
specifying different targets for the possible tasks.

"""

import copy

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from mantra.manifold_types import Manifold2Type

NAME_TO_CLASS_2M = {
    manifold.value: index for index, manifold in enumerate(Manifold2Type)
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


class NameToClass2MTransform(T.BaseTransform):
    """
    Encode the homemorphism type (`name`) as a nominal target for 2-manifolds.
    """

    def __init__(self):
        super().__init__()
        self.class_dict = NAME_TO_CLASS_2M

    def forward(self, data: Data):
        assert "name" in data
        if data.name not in self.class_dict:
            raise KeyError(
                f"Unknown 2-manifold name {data.name!r}; expected one of "
                f"{sorted(self.class_dict)}."
            )
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

class CreateLabels(BaseTransform):
    """Create labels based on attributes.

    This transform creates labels based on attributes present in
    a dataset. Depending on the type of attribute, labels may be
    binary or multi-class.
    """

    def __init__(self, source):
        """Create new label creator transform.

        Parameters
        ----------
        source : str
            Denotes attribute that is used to create labels. If not
            present in the data, the `forward()` function will just
            fail with an exception.
        """
        super().__init__()

        self.source = source
        self.label_to_index = {}
        self.index_remap = {}

    def _assign_precompute(self, data):
        assert (
            self.source in data
        ), f"Source attribute '{self.source}' is not present in data"

        label = data[self.source]

        if isinstance(label, bool):
            # Booleans map directly: ``False = 0`` and ``True = 1``.
            data.y = torch.tensor([int(label)])
        else:
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label not in self.label_to_index:
                self.label_to_index[label] = self.index_remap[label] = len(
                    self.label_to_index
                )
            data.y = torch.tensor([self.label_to_index[label]])

        data.label = label

        return data

    def forward(self, data):
        """Assign label for a given `data` object.

        Given a source attribute to create a label, assign a numerical
        index to be used for downstream classification tasks. There is
        one interesting thing happening here: The class assigns labels
        based on the data type. If a boolean property is detected, the
        mapping will default to `False = 0` and `True = 1`. Otherwise,
        for string-based attributes, indices will be assigned based on
        the order in which they are encountered.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object. The source attribute, which is used to
            create labels, must be present.

        Returns
        -------
        torch_geometric.data.Data
            Data object with a label attached to it, stored in the `y`
            attribute of the tensor.
        """
        # In this case, we are performing a remapping in either
        # 1) Dataset was already preprocessed but we filtered,
        # or, 2) we are loading the dataset
        if "label" in data:
            remap = copy.copy(data.y.item())

            if remap not in self.index_remap:
                self.index_remap[remap] = len(self.index_remap)

            data.y = torch.tensor([self.index_remap[remap]])
        else:
            data = self._assign_precompute(data)

        return data
