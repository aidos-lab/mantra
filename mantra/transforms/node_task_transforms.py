"""Node-level task transforms module

A set of transforms for MANTRA that specify a node-level prediction
target (`data.y`) from an attribute holding one value per vertex of
the triangulation.

Like the graph-level task transforms, these transforms are stateless:
the target of a sample is a pure function of its stored attributes.
Per-vertex attributes must be stored as 1-D tensors.

In addition to `data.y`, both transforms store a boolean
`data.node_mask` of shape `(n_vertices,)` that selects the supervised
vertices, so that losses and metrics can be restricted to them.
"""

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data


class _NodeAttributeTransform(T.BaseTransform):
    """Shared validation of a per-vertex attribute and node masking."""

    def __init__(self, source, mask_first=False):
        super().__init__()

        self.source = source
        self.mask_first = mask_first

    def _node_values(self, data: Data):
        """Return the per-vertex values of `source` as a 1-D tensor."""
        assert (
            self.source in data
        ), f"Source attribute '{self.source}' is not present in data"

        values = data[self.source]

        assert isinstance(
            values, torch.Tensor
        ), f"Attribute '{self.source}' must be a tensor, got {type(values).__name__}"
        assert values.dim() == 1, (
            f"Attribute '{self.source}' must hold one value per vertex, "
            f"got shape {tuple(values.shape)}"
        )

        if "n_vertices" in data:
            n_vertices = int(data.n_vertices)
            assert values.numel() == n_vertices, (
                f"Attribute '{self.source}' holds {values.numel()} values "
                f"for {n_vertices} vertices"
            )

        return values

    def _node_mask(self, n_vertices):
        """Boolean mask of the supervised vertices."""
        mask = torch.ones(n_vertices, dtype=torch.bool)
        if self.mask_first:
            mask[0] = False
        return mask


class AttributeToNodeRegressionTransform(_NodeAttributeTransform):
    """Encode a per-vertex attribute as a node-level regression target.

    The values are stored in `data.y` as a float tensor of shape
    `(n_vertices, 1)`; `data.node_mask` selects the supervised
    vertices.
    """

    def __init__(self, source, mask_first=False):
        """Create a new node-level regression-target transform.

        Parameters
        ----------
        source : str
            Per-vertex attribute used as the target. Must be present in
            the data as a 1-D tensor with one value per vertex.

        mask_first : bool
            If set, the first vertex is excluded from supervision via
            `node_mask`. This is meant for vertices that carry no
            target of their own, such as the origin of the polytope in
            a Calabi-Yau triangulation, which corresponds to no toric
            divisor.
        """
        super().__init__(source, mask_first)

    def forward(self, data: Data):
        values = self._node_values(data)

        data.y = values.to(torch.float32).view(-1, 1)
        data.node_mask = self._node_mask(values.numel())
        return data


class AttributeToNodeClassTransform(_NodeAttributeTransform):
    """Encode a per-vertex attribute as node-level class indices.

    Every vertex value is passed through a fixed `mapping` from
    attribute values to class indices, so that the same value always
    yields the same index. The indices are stored in `data.y` as a
    `long` tensor of shape `(n_vertices,)`; `data.node_mask` selects
    the supervised vertices.
    """

    def __init__(self, source, mapping, mask_first=False):
        """Create a new node-level class-index transform.

        Parameters
        ----------
        source : str
            Per-vertex attribute used as the label. Must be present in
            the data as a 1-D tensor with one value per vertex.

        mapping : dict
            Fixed mapping from attribute values to class indices.

        mask_first : bool
            If set, the first vertex is excluded from supervision via
            `node_mask`; see `AttributeToNodeRegressionTransform`.
        """
        super().__init__(source, mask_first)

        self.mapping = dict(mapping)

    @property
    def num_classes(self):
        """Number of classes of the mapping."""
        return len(self.mapping)

    def forward(self, data: Data):
        values = self._node_values(data)

        data.y = torch.tensor(
            [self.mapping[value] for value in values.tolist()],
            dtype=torch.long,
        )
        data.node_mask = self._node_mask(values.numel())
        return data
