"""Per-triangle targets from the sparse intersection numbers of a
Calabi-Yau triangulation."""

from itertools import combinations

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class TriangleIntersectionNumbers(BaseTransform):
    """Write the intersection number of three toric divisors per boundary
    triangle.

    `source` holds rows `(i, j, k, value)` over the triangulation points
    (0-based). Point 0 is the origin, which CYTools identifies with the
    canonical class `K = -sum_l D_l`, so rows with index 0 are not
    intersections of three toric divisors and are ignored, like the rows
    with repeated indices (self-intersections). The remaining rows lie on
    the boundary triangles of the triangulation, since three divisors
    meet only if their cone is in the fan.

    Parameters
    ----------
    source : str
        Attribute holding the sparse intersection numbers.
    target : str
        Receives one value per boundary triangle (the triangles without
        the origin), zero where `source` has no row, in the lexicographic
        order of the triangles.
    index : str
        Receives the `(3, n_boundary_triangles)` matrix of the triangles'
        0-based vertices in the same order, so that vertex embeddings can
        be gathered per triangle.
    mask : str
        Receives one boolean per triangle of the triangulation, in the
        lexicographic order that the simplicial representations and
        `PropagateConvexComb` use for rank 2, true on the boundary
        triangles, so that one state per triangle of the complex can be
        restricted to the rows of `target`.
    """

    def __init__(
        self,
        source: str = "intersection_numbers",
        target: str = "kappa_2",
        index: str = "simplex_index_2",
        mask: str = "triangle_mask",
    ):
        super().__init__()
        self.source = source
        self.target = target
        self.index = index
        self.mask = mask

    def forward(self, data: Data) -> Data:
        coords = getattr(data, "coords", None)
        if coords is not None and torch.as_tensor(coords)[0].any():
            raise ValueError("point 0 must be the origin")
        triangles = sorted(
            {
                tuple(v - 1 for v in face)
                for simplex in data.triangulation
                for face in combinations(sorted(simplex), 3)
            }
        )
        boundary = [t for t in triangles if t[0] != 0]
        if self.source not in data:
            raise ValueError(
                f"`{self.source}` is missing; the parquet must carry the "
                "sparse intersection numbers"
            )
        source = torch.as_tensor(data[self.source]).reshape(-1, 4)
        values = {}
        for *idx, value in source.tolist():
            if len(set(idx)) == 3 and 0 not in idx:
                values[tuple(sorted(idx))] = value
        if not values.keys() <= set(boundary):
            raise ValueError(
                f"`{self.source}` has entries outside the triangulation"
            )
        data[self.target] = torch.tensor(
            [values.get(t, 0) for t in boundary], dtype=torch.float32
        )
        data[self.index] = (
            torch.tensor(boundary, dtype=torch.long).reshape(-1, 3).t()
        )
        data[self.mask] = torch.tensor([t[0] != 0 for t in triangles])
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(source={self.source!r}, "
            f"target={self.target!r}, index={self.index!r}, "
            f"mask={self.mask!r})"
        )
