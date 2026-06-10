import warnings

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data


class TriangulationToFaceTransform:
    """
    Populate ``data.face`` (PyG's ``[3, M]`` triangle representation) from
    ``data.triangulation``.

    .. deprecated::
        This transform is deprecated and is no longer wired into the dataset
        factory. For 2-manifolds it produces a genuine triangle mesh, but for
        3-manifolds ``data.face`` is the complex's *full 2-skeleton* (every
        2-face of the complex after dedup), which is not an embedded surface
        and therefore does not match PyG's mesh ``face`` semantics. No model
        currently consumes ``data.face``. Use the ``one_skeleton``, ``dual``,
        or ``hasse`` representations instead. The class is retained only for
        backwards compatibility and will be removed in a future release.

    Accepts either a 2-manifold triangulation (list of length-3 simplices,
    shape ``[N, 3]``) or a 3-manifold triangulation (list of length-4
    tetrahedra, shape ``[N, 4]``). For 2-manifolds the top simplices already
    are triangles. For 3-manifolds the four triangular boundary faces are
    enumerated per tetrahedron and triangles shared between tets are
    deduplicated regardless of vertex order.

    MANTRA stores vertex IDs as 1-based (LEX convention); the resulting
    ``data.face`` uses 0-based indices, matching the PyG convention.
    """

    def __init__(self, remove_triangulation: bool = False) -> None:
        warnings.warn(
            "TriangulationToFaceTransform is deprecated and is no longer used "
            "by the dataset factory. For 3-manifolds `data.face` is the full "
            "2-skeleton, not a surface mesh. Use the 'one_skeleton', 'dual', "
            "or 'hasse' representations instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.remove_triangulation = remove_triangulation

    def __call__(self, data):
        if not hasattr(data, "triangulation") or data.triangulation is None:
            return data

        tri = torch.tensor(data.triangulation, dtype=torch.long)
        assert tri.ndim == 2, f"triangulation must be 2-d, got {tri.shape}"
        K = tri.shape[1]

        if K == 3:
            face = tri.T
        elif K == 4:
            idx = torch.tensor(
                [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            )
            face = tri[:, idx].reshape(-1, 3).T
        else:
            raise ValueError(
                f"Unsupported simplex size K={K} (expected 3 or 4)"
            )

        face = face - 1

        face_sorted, _ = face.sort(dim=0)
        _, inv = torch.unique(face_sorted, dim=1, return_inverse=True)
        n_classes = int(inv.max()) + 1
        first_occurrence = torch.full((n_classes,), -1, dtype=torch.long)
        for col, cls in enumerate(inv.tolist()):
            if first_occurrence[cls] == -1:
                first_occurrence[cls] = col
        data.face = face[:, first_occurrence]

        if self.remove_triangulation:
            data.triangulation = None
        else:
            data.triangulation = tri

        return data


class SetNumNodesTransform(T.BaseTransform):
    """
    Convert the `n_vertices` attribute to `num_nodes`.
    """

    def forward(self, data: Data):
        assert "n_vertices" in data
        data.num_nodes = data.n_vertices
        return data
