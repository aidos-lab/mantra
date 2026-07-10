import torch
from torch_geometric.transforms import BaseTransform


class ScalarFeatures(BaseTransform):
    """Collect scalar attributes into a single feature vector.

    This transform assembles per-sample scalar attributes (e.g. the
    lattice-point counts of a Calabi-Yau triangulation) into a feature
    tensor of shape `(1, k)`, providing a graph-level input for
    baseline models that do not consume the triangulation itself.
    """

    def __init__(self, sources):
        """Create new scalar feature transform.

        Parameters
        ----------
        sources : list of str
            Scalar attributes to collect, in order. Each attribute
            must be present in the data.
        """
        super().__init__()

        self.sources = [sources] if isinstance(sources, str) else list(sources)

    def forward(self, data):
        """Assign scalar feature vector for a given `data` object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object. All source attributes must be present.

        Returns
        -------
        torch_geometric.data.Data
            Data object with a new `scalar_features` key of shape
            `(1, k)`.
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

        data.scalar_features = torch.tensor(
            values, dtype=torch.float32
        ).view(1, -1)
        return data
