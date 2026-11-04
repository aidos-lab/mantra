"""Per-triangle targets from the sparse intersection numbers of a
Calabi-Yau triangulation."""

from itertools import combinations

import numpy as np
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
        values = _divisor_intersections(data, self.source, boundary)
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


def _divisor_intersections(data, source, boundary):
    """`{(i, j, k): value}` over the boundary triangles of `data`."""
    if source not in data:
        raise ValueError(
            f"`{source}` is missing; the parquet must carry the sparse "
            "intersection numbers"
        )
    values = {}
    for *idx, value in torch.as_tensor(data[source]).reshape(-1, 4).tolist():
        if len(set(idx)) == 3 and 0 not in idx:
            values[tuple(sorted(idx))] = value
    if not values.keys() <= set(boundary):
        raise ValueError(f"`{source}` has entries outside the triangulation")
    return values


class CandidateTriangleIntersectionNumbers(BaseTransform):
    """Write the intersection number per candidate triangle of the
    polytope's 2-faces, zero where the triangulation has no such triangle.

    The candidates are the point triples on a common 2-face with at
    least `min_points` lattice points. The fine regular star
    triangulations of a polytope differ only in how they triangulate
    such 2-faces (that choice is the Calabi-Yau phase), so a candidate's
    target depends on the triangulation: a present triple carries the
    intersection number of its three divisors, an absent one zero. The
    triples of a triangular 2-face are present in every fine
    triangulation, which is why the default `min_points` leaves them out.
    Collinear triples (three points on an edge) are never triangles and
    are dropped when `coords` is available.

    Parameters
    ----------
    source : str
        Attribute holding the sparse intersection numbers, rows
        `(i, j, k, value)` over the 0-based triangulation points.
    faces : str
        Attribute holding the lattice points of the 2-faces of the
        polytope as rows `(face, point)` of 0-based indices.
    target : str
        Receives one value per candidate, in the lexicographic order of
        the candidates.
    index : str
        Receives the `(3, n_candidates)` matrix of the candidates'
        vertices in the same order.
    present : str
        Receives one boolean per candidate, true when the triple is a
        triangle of the triangulation.
    min_points : int
        2-faces with fewer lattice points contribute no candidates.
    """

    def __init__(
        self,
        source: str = "intersection_numbers",
        faces: str = "face2_points",
        target: str = "kappa_candidate",
        index: str = "candidate_index_2",
        present: str = "candidate_present",
        min_points: int = 4,
    ):
        super().__init__()
        self.source = source
        self.faces = faces
        self.target = target
        self.index = index
        self.present = present
        self.min_points = min_points

    def forward(self, data: Data) -> Data:
        coords = getattr(data, "coords", None)
        if coords is not None:
            coords = np.asarray(torch.as_tensor(coords).tolist())
            if coords[0].any():
                raise ValueError("point 0 must be the origin")
        if self.faces not in data:
            raise ValueError(
                f"`{self.faces}` is missing; the parquet must carry the "
                "2-faces of the polytope (make_phases_parquet.py)"
            )
        triangles = sorted(
            {
                tuple(v - 1 for v in face)
                for simplex in data.triangulation
                for face in combinations(sorted(simplex), 3)
            }
        )
        boundary = [t for t in triangles if t[0] != 0]
        values = _divisor_intersections(data, self.source, boundary)
        faces = {}
        for face, point in (
            torch.as_tensor(data[self.faces]).reshape(-1, 2).tolist()
        ):
            faces.setdefault(face, []).append(point)
        candidates = set()
        for face in faces.values():
            if 0 in face:
                raise ValueError("a 2-face contains the origin")
            if len(face) >= self.min_points:
                candidates.update(combinations(sorted(face), 3))
        if coords is not None:
            candidates = {
                t
                for t in candidates
                if np.linalg.matrix_rank(coords[list(t[1:])] - coords[t[0]])
                == 2
            }
        candidates = sorted(candidates)
        present = set(boundary)
        data[self.target] = torch.tensor(
            [values.get(t, 0) if t in present else 0 for t in candidates],
            dtype=torch.float32,
        )
        data[self.index] = (
            torch.tensor(candidates, dtype=torch.long).reshape(-1, 3).t()
        )
        data[self.present] = torch.tensor(
            [t in present for t in candidates], dtype=torch.bool
        )
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(source={self.source!r}, "
            f"faces={self.faces!r}, target={self.target!r}, "
            f"index={self.index!r}, present={self.present!r}, "
            f"min_points={self.min_points})"
        )
