import copy

import torch
from torch_geometric.transforms import BaseTransform


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
        # Whenever the source attribute is available, (re)build the
        # label mapping from it. This keeps `label_to_index` consistent
        # regardless of whether the data was freshly processed, loaded
        # from a cache, or filtered in between: surviving labels are
        # indexed compactly in order of appearance.
        if self.source in data or "y" not in data:
            data = self._assign_precompute(data)
        else:
            # Fallback for preprocessed data that only carries `y`:
            # remap the existing indices compactly.
            remap = copy.copy(data.y.item())

            if remap not in self.index_remap:
                self.index_remap[remap] = len(self.index_remap)

            data.y = torch.tensor([self.index_remap[remap]])

        return data


class CreateRegressionLabels(BaseTransform):
    """Create regression targets based on attributes.

    This transform assembles a float target vector `y` from one or
    more scalar attributes present in a dataset, e.g. Hodge numbers
    of a Calabi-Yau triangulation. In contrast to `CreateLabels`, the
    attribute values are used directly instead of being mapped to
    class indices.
    """

    def __init__(self, sources, sum_sources=False):
        """Create new regression label transform.

        Parameters
        ----------
        sources : str or list of str
            Attribute(s) used to create the target. Each attribute must
            be a scalar that is present in the data.

        sum_sources : bool
            If set, the values of all sources are summed into a single
            scalar target (e.g. `h11 + h12`) instead of being stacked
            into a vector.
        """
        super().__init__()

        self.sources = [sources] if isinstance(sources, str) else list(sources)
        self.sum_sources = sum_sources

        # Kept for interface compatibility with `CreateLabels`; there
        # is no label indexing for regression targets.
        self.label_to_index = {}

    def forward(self, data):
        """Assign regression target for a given `data` object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object. All source attributes must be present.

        Returns
        -------
        torch_geometric.data.Data
            Data object with the target attached to it, stored in the
            `y` attribute as a float tensor of shape `(1, k)`, with
            `k` the number of sources (or `1` if `sum_sources`).
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

        y = torch.tensor(values, dtype=torch.float32).view(1, -1)

        if self.sum_sources:
            y = y.sum(dim=1, keepdim=True)

        data.y = y
        return data
