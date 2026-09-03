"""Task transforms module

A set of transforms for MANTRA that serve the purpose of specifying
the prediction target (`data.y`) of the possible tasks.

All transforms in this module are stateless: a sample always
receives the same target regardless of the order in which, or the
subset with which, the dataset is traversed. Class indices come from
fixed mappings: `NAME_TO_CLASS_2M` and `NAME_TO_CLASS_3M` for the
homeomorphism types, or a caller-supplied `{value: index}` mapping
for any other attribute (for integer-valued attributes such as
`genus`, build it once from the values present in the full dataset).
Remapping these canonical indices to a contiguous range over the
classes present in a particular training split needs to be performed
in the training code.
"""

from typing import Dict

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

from mantra.manifold_types import Manifold2Type, Manifold3Type

NAME_TO_CLASS_2M = {
    manifold.value: index for index, manifold in enumerate(Manifold2Type)
}
"""Canonical class index of every 2-manifold homeomorphism type."""

NAME_TO_CLASS_3M = {
    manifold.value: index for index, manifold in enumerate(Manifold3Type)
}
"""Canonical class index of every 3-manifold homeomorphism type."""


class AttributeToClassTransform(T.BaseTransform):
    """Encode a stored attribute as a class index in `data.y`.

    The class index is looked up in a fixed `mapping` from attribute
    values to indices, so the same attribute value always yields the
    same index, independent of the samples seen before. Scalar
    attributes arrive as Python values when the transform runs as a
    `pre_transform` and as one-element integer tensors after the
    dataset has been collated; tensors are unwrapped before the lookup.
    """

    # Used in error messages; subclasses override it with a more
    # specific description of the attribute.
    _value_description = "value"

    def __init__(self, source, mapping: Dict):
        """Create a new class-index transform.

        Parameters
        ----------
        source : str
            Attribute used as the label. Must be present in the data.

        mapping : Dict
            Fixed mapping from attribute values to class indices, e.g.
            `NAME_TO_CLASS_2M`. For integer-valued attributes such as
            `genus`, build it once from the values present in the full
            dataset, e.g. `{v: i for i, v in enumerate(sorted(values))}`,
            so that it does not depend on a split or traversal order.
        """
        super().__init__()

        self.source = source
        self.mapping = mapping

    @property
    def num_classes(self):
        """Number of classes of the mapping."""
        return len(self.mapping)

    def forward(self, data: Data):
        assert (
            self.source in data
        ), f"Source attribute '{self.source}' is not present in data"

        value = data[self.source]

        if isinstance(value, torch.Tensor):
            assert len(value.shape) == 0 or (
                len(value.shape) == 1 and value.shape[0] == 1
            ), "Needs to be a 1 element tensor, i.e. scalar"
            assert not torch.is_floating_point(
                value
            ), "Tensor needs to be of type int"
            value = value.item()

        if self.mapping is None:
            index = value
        elif value not in self.mapping:
            raise KeyError(
                f"Unknown {self._value_description} {value!r}; "
                f"expected one of {sorted(self.mapping, key=str)}."
            )
        else:
            index = self.mapping[value]

        data.y = torch.tensor(index, dtype=torch.long)
        data.label = value

        return data


class NameToClass2MTransform(AttributeToClassTransform):
    """
    Encode the homeomorphism type (`name`) of a 2-manifold as a class
    index using `NAME_TO_CLASS_2M`.
    """

    _value_description = "2-manifold name"

    def __init__(self):
        super().__init__(source="name", mapping=NAME_TO_CLASS_2M)


class NameToClass3MTransform(AttributeToClassTransform):
    """
    Encode the homeomorphism type (`name`) of a 3-manifold as a class
    index using `NAME_TO_CLASS_3M`.
    """

    _value_description = "3-manifold name"

    def __init__(self):
        super().__init__(source="name", mapping=NAME_TO_CLASS_3M)


class OrientableToClassTransform(T.BaseTransform):
    """
    Encode the orientability target as a binary target
    with type `long`.
    """

    def forward(self, data: Data):
        data.orientable = torch.tensor(data.betti_numbers)[..., -1]
        data.y = data.orientable.long()
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


class AttributeToRegressionTransform(T.BaseTransform):
    """Assemble a float regression target from a stored attribute.

    The target `y` is the attribute `source`, a scalar such as `genus`
    or `n_vertices` or a fixed-length vector such as `betti_numbers`,
    converted to a float tensor of shape `(1, k)`. Attribute values
    are used directly, so the target of a sample never depends on
    other samples.
    """

    def __init__(self, source: str):
        """Create a new regression-target transform.

        Parameters
        ----------
        source : str
            Attribute used as the target. Must be a scalar or a
            fixed-length vector present in the data.
        """
        super().__init__()
        self.source = source

    def forward(self, data: Data):
        """Assign the regression target of a given `data` object.

        Returns
        -------
        torch_geometric.data.Data
            Data object with the target stored in `y` as a float tensor
            of shape `(1, k)`, with `k` the number of elements of the
            attribute (`1` for scalars).
        """
        # `as_tensor` accepts Python scalars and lists (`pre_transform`
        # path) as well as the one-element tensors produced by the
        # collated dataset (`transform` path); `reshape` gives the same
        # `(1, k)` shape on both paths.
        data.y = torch.as_tensor(
            getattr(data, self.source), dtype=torch.float32
        ).reshape(1, -1)
        return data
