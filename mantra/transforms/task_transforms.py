"""Task transforms module

A set of transforms for MANTRA that serve the purpose of specifying
the prediction target (`data.y`) of the possible tasks.

All transforms in this module are stateless: a sample always
receives the same target regardless of the order in which, or the
subset with which, the dataset is traversed. Class indices of
categorical attributes come from fixed mappings such as
`NAME_TO_CLASS_2M`; integer-valued attributes are used as class
indices directly. Remapping these canonical indices to a contiguous
range over the classes present in a particular training split needs 
to be performed in the training code.
"""

import numbers

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


def _as_scalar(value):
    """Unwrap a one-element tensor into a Python scalar.

    Non-tensor values (Python or numpy scalars, strings) are returned
    unchanged.
    """
    if isinstance(value, torch.Tensor):
        assert (
            value.numel() == 1
        ), f"Expected a scalar attribute, got {value.numel()} elements"
        return value.item()
    return value


class AttributeToClassTransform(T.BaseTransform):
    """Encode a stored attribute as a class index in `data.y`.

    The class index is determined statically: either through a fixed
    `mapping` from attribute values to indices or, for integer-valued
    attributes, by using the value itself as the index. Either way the
    same attribute value always yields the same index, independent of
    the samples seen before.
    """

    # Used in error messages; subclasses override it with a more
    # specific description of the attribute.
    _value_description = "value"

    def __init__(self, source, mapping=None):
        """Create a new class-index transform.

        Parameters
        ----------
        source : str
            Attribute used as the label. Must be present in the data.

        mapping : dict or None
            Fixed mapping from attribute values to class indices, e.g.
            `NAME_TO_CLASS_2M`. If `None`, the attribute must be
            integer-valued (`int`, `bool` or a one-element integer
            tensor) and is used as the class index directly.
        """
        super().__init__()

        self.source = source
        self.mapping = None if mapping is None else dict(mapping)

    @property
    def num_classes(self):
        """Number of classes of the mapping; `None` without a mapping."""
        return None if self.mapping is None else len(self.mapping)

    def forward(self, data: Data):
        assert (
            self.source in data
        ), f"Source attribute '{self.source}' is not present in data"

        value = _as_scalar(data[self.source])

        if self.mapping is None:
            if not isinstance(value, numbers.Integral):
                raise TypeError(
                    f"Attribute '{self.source}' must be integer-valued to "
                    f"serve as a class index, got {type(value).__name__}; "
                    "pass an explicit `mapping` instead."
                )
            index = int(value)
        else:
            if value not in self.mapping:
                raise KeyError(
                    f"Unknown {self._value_description} {value!r}; "
                    f"expected one of {sorted(self.mapping, key=str)}."
                )
            index = self.mapping[value]

        data.y = torch.tensor(index, dtype=torch.long)
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
    """Assemble a float regression target from scalar attributes.

    The target vector `y` is built from one or more scalar attributes
    present in a sample, e.g. the Hodge numbers of a Calabi-Yau
    triangulation. Attribute values are used directly, so the target
    of a sample never depends on other samples.
    """

    def __init__(self, sources, sum_sources=False):
        """Create a new regression-target transform.

        Parameters
        ----------
        sources : str or list of str
            Attribute(s) used to build the target. Each attribute must
            be a scalar present in the data.

        sum_sources : bool
            If set, the values of all sources are summed into a single
            scalar target (e.g. `h11 + h12`) instead of being stacked
            into a vector.
        """
        super().__init__()

        self.sources = [sources] if isinstance(sources, str) else list(sources)
        self.sum_sources = sum_sources

    def forward(self, data: Data):
        """Assign the regression target of a given `data` object.

        Returns
        -------
        torch_geometric.data.Data
            Data object with the target stored in `y` as a float tensor
            of shape `(1, k)`, with `k` the number of sources (or `1`
            if `sum_sources`).
        """
        values = []
        for source in self.sources:
            assert (
                source in data
            ), f"Source attribute '{source}' is not present in data"
            values.append(float(_as_scalar(data[source])))

        y = torch.tensor(values, dtype=torch.float32).view(1, -1)

        if self.sum_sources:
            y = y.sum(dim=1, keepdim=True)

        data.y = y
        return data
