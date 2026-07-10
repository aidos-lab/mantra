import numpy as np
import torch
from torch_geometric.transforms import BaseTransform

from mantra.transforms.moment_curve_embedding import _propagate_values


class CoordinateEmbedding(BaseTransform):
    """Expose stored vertex coordinates as a feature embedding.

    In contrast to :class:`MomentCurveEmbedding`, which synthesizes
    coordinates from the number of vertices, this transform uses *real*
    vertex coordinates stored in the `vertices` attribute of a `Data`
    object, e.g. the lattice coordinates of a Calabi-Yau triangulation.
    """

    def __init__(self, propagate=False, append_attributes=None):
        """Create new coordinate embedding transform.

        Parameters
        ----------
        propagate : bool
            If set, propagates the coordinates from the 0-simplices to
            all higher simplices by calculating barycenters, resulting
            in a dictionary keyed by simplex dimension. This requires
            `triangulation` to be present in the data object. If not
            set, the plain vertex coordinate tensor is used.

        append_attributes : list of str or None
            Scalar attributes of the data object (e.g. lattice-point
            counts) to broadcast as additional constant feature
            columns. Since barycenters of constants are constant, the
            attributes reach every simplex rank when propagating.
        """
        super().__init__()

        self.propagate = propagate
        self.append_attributes = (
            list(append_attributes) if append_attributes else []
        )

    def forward(self, data):
        """Assign coordinate embedding for a data object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object. Must contain a `vertices` attribute of
            shape `(n_vertices, coord_dim)`; row `i` holds the
            coordinates of (1-indexed) vertex `i + 1`.

        Returns
        -------
        torch_geometric.data.Data
            Data object with a new `coordinate_embedding` key added.
            The attribute will be overwritten if already present.
        """
        assert "vertices" in data, "Data object must contain `vertices`"

        X = data["vertices"]
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float32)

        for attribute in self.append_attributes:
            assert (
                attribute in data
            ), f"Attribute '{attribute}' is not present in data"

            value = data[attribute]
            if isinstance(value, torch.Tensor):
                value = value.item()

            X = np.column_stack(
                [X, np.full((X.shape[0], 1), float(value), dtype=np.float32)]
            )

        if self.propagate:
            assert (
                "triangulation" in data
            ), "Data object must contain `triangulation` to perform propagation"
            data["coordinate_embedding"] = _propagate_values(
                X, data["triangulation"]
            )
        else:
            data["coordinate_embedding"] = torch.from_numpy(X).to(
                torch.float32
            )

        return data
