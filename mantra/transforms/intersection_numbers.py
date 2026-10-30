"""Per-triangle targets from the sparse intersection numbers of a
Calabi-Yau triangulation."""

from itertools import combinations

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class TriangleIntersectionNumbers(BaseTransform):
    """Write the intersection numbers of three distinct divisors per triangle.

    `source` holds rows `(i, j, k, value)` over the triangulation points
    (0-based, the origin included). The entries with three distinct
    indices lie on the triangles of the triangulation, since three
    divisors meet only if their cone is in the fan; entries with
    repeated indices (self-intersections) are ignored.

    Parameters
    ----------
    source : str
        Attribute holding the sparse intersection numbers.
    target : str
        Receives one value per triangle, zero where `source` has no
        entry, in the lexicographic order of the triangles that the
        simplicial representations and `PropagateConvexComb` use for
        rank 2.
    index : str
        Receives the `(3, n_triangles)` matrix of the triangles' 0-based
        vertices in the same order, so that vertex embeddings can be
        gathered per triangle.
    """

    def __init__(
        self,
        source: str = "intersection_numbers",
        target: str = "kappa_2",
        index: str = "simplex_index_2",
    ):
        super().__init__()
        self.source = source
        self.target = target
        self.index = index

    def forward(self, data: Data) -> Data:
        triangles = sorted(
            {
                tuple(v - 1 for v in face)
                for simplex in data.triangulation
                for face in combinations(sorted(simplex), 3)
            }
        )
        source = torch.as_tensor(data[self.source]).reshape(-1, 4)
        values = {}
        for *idx, value in source.tolist():
            if len(set(idx)) == 3:
                values[tuple(sorted(idx))] = value
        if not values.keys() <= set(triangles):
            raise ValueError(
                f"`{self.source}` has entries outside the triangulation"
            )
        data[self.target] = torch.tensor(
            [values.get(t, 0) for t in triangles], dtype=torch.float32
        )
        data[self.index] = (
            torch.tensor(triangles, dtype=torch.long).reshape(-1, 3).t()
        )
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(source={self.source!r}, "
            f"target={self.target!r}, index={self.index!r})"
        )
