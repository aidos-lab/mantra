"""Node-level task transforms module

A set of transforms for MANTRA that specify a node-level prediction
target (`data.y`) from an attribute holding one value per vertex of
the triangulation.

Like the graph-level task transforms, these transforms are stateless:
the target of a sample is a pure function of its stored attributes.
The per-vertex attribute may be a 1-D tensor (the `transform` path,
after the dataset has been collated) or a list of scalars (the
`pre_transform` path).
"""

from typing import Dict

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data


def _node_values(data: Data, source: str) -> torch.Tensor:
    """Return the per-vertex values of `source` as a 1-D tensor."""
    values = torch.as_tensor(data[source])

    # A `(n, k)` attribute would silently yield n * k targets.
    assert values.dim() == 1, (
        f"Attribute '{source}' must hold one value per vertex, "
        f"got shape {tuple(values.shape)}"
    )

    return values


class AttributeToNodeRegressionTransform(T.BaseTransform):
    """Encode a per-vertex attribute as a node-level regression target.

    The values are stored in `data.y` as a float tensor of shape
    `(n_vertices, 1)`, one row per vertex in vertex order.
    """

    def __init__(self, source: str):
        """Create a new node-level regression-target transform.

        Parameters
        ----------
        source : str
            Per-vertex attribute used as the target. Must be present in
            the data with one value per vertex.
        """
        super().__init__()

        self.source = source

    def forward(self, data: Data):
        values = _node_values(data, self.source)

        data.y = values.to(torch.float32).view(-1, 1)
        return data


class AttributeToNodeClassTransform(T.BaseTransform):
    """Encode a per-vertex attribute as node-level class indices.

    Every vertex value is passed through a fixed `mapping` from
    attribute values to class indices, so that the same value always
    yields the same index. The indices are stored in `data.y` as a
    `long` tensor of shape `(n_vertices,)`, one entry per vertex in
    vertex order.
    """

    def __init__(self, source: str, mapping: Dict):
        """Create a new node-level class-index transform.

        Parameters
        ----------
        source : str
            Per-vertex attribute used as the label. Must be present in
            the data with one integer value per vertex.

        mapping : Dict
            Fixed mapping from attribute values to class indices. As
            for `AttributeToClassTransform`, build it once from the
            values present in the full dataset, so that it does not
            depend on a split or traversal order.
        """
        super().__init__()

        self.source = source
        self.mapping = mapping

    @property
    def num_classes(self):
        """Number of classes of the mapping."""
        return len(self.mapping)

    def forward(self, data: Data):
        values = _node_values(data, self.source)

        assert not torch.is_floating_point(
            values
        ), "Tensor needs to be of type int"

        indices = []
        for value in values.tolist():
            if value not in self.mapping:
                raise KeyError(
                    f"Unknown value {value!r}; "
                    f"expected one of {sorted(self.mapping, key=str)}."
                )
            indices.append(self.mapping[value])

        data.y = torch.tensor(indices, dtype=torch.long)
        return data
