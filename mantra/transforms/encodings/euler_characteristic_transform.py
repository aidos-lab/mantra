"""Euler characteristic transform of a geometrically realised triangulation."""

from itertools import combinations
from typing import Optional

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform


class EulerCharacteristicTransform(BaseTransform):
    """Euler characteristic transform (ECT) of a triangulation with coordinates.

    For every direction ``v`` and threshold ``t`` the ECT is the Euler
    characteristic of the subcomplex spanned by the vertices whose
    height ``<x, v>`` is at most ``t``: the alternating sum over ranks
    of the number of simplices whose highest vertex lies below ``t``.
    This is the sublevel-set ECT of the piecewise-linear height
    function; see Turner, Mukherjee and Boyer (2014), "Persistent
    homology transform for modeling shapes and surfaces".

    The directions are drawn once, uniformly on the unit sphere of the
    coordinate space, from a generator seeded with ``seed``, so every
    sample of a dataset is transformed along the same directions and
    the direction index is meaningful across samples. Thresholds are
    ``resolution`` equally spaced values in ``[-radius, radius]``;
    simplices above ``radius`` are never counted.

    Parameters
    ----------
    num_directions : int
        Number of directions.
    resolution : int
        Number of thresholds per direction.
    radius : float
        Half-width of the threshold interval. Choose it to cover the
        vertex norms of the dataset.
    max_rank : int or None
        Compute the ECT of the ``max_rank``-skeleton, e.g. ``1`` for the
        one-skeleton (vertices and edges). ``None`` uses every face of
        the triangulation.
    per_rank : bool
        Keep the face counts of each rank apart instead of taking
        their alternating sum: the result then holds, per direction
        and threshold, the number of vertices, edges, ... below the
        threshold (the f-vector of the sublevel subcomplex). The
        alternating sum of the ranks is the ECT, so this carries
        strictly more information.
    source : str
        Attribute holding one coordinate row per vertex, in vertex
        order (vertex ``i`` of the 1-indexed triangulation is row
        ``i - 1``).
    target : str
        Attribute to write.
    seed : int
        Seed of the direction generator.

    Notes
    -----
    The result has shape ``(1, num_directions, resolution)`` and dtype
    ``torch.int16``, so that PyG batches it along the leading axis into
    ``(batch_size, num_directions, resolution)``; convert it to a float
    tensor before feeding it to a model. With ``per_rank`` the shape is
    ``(1, max_rank + 1, num_directions, resolution)``.
    """

    def __init__(
        self,
        num_directions: int = 64,
        resolution: int = 64,
        radius: float = 1.0,
        max_rank: Optional[int] = None,
        per_rank: bool = False,
        source: str = "coords",
        target: str = "ect",
        seed: int = 0,
    ):
        super().__init__()
        self.num_directions = num_directions
        self.resolution = resolution
        self.radius = radius
        self.max_rank = max_rank
        self.per_rank = per_rank
        self.source = source
        self.target = target
        self.seed = seed
        self._directions = None

    def directions(self, ambient_dim: int) -> torch.Tensor:
        """The ``(ambient_dim, num_directions)`` unit directions."""
        if (
            self._directions is None
            or self._directions.shape[0] != ambient_dim
        ):
            rng = np.random.default_rng(self.seed)
            v = rng.standard_normal((ambient_dim, self.num_directions))
            v /= np.linalg.norm(v, axis=0, keepdims=True)
            self._directions = torch.tensor(v, dtype=torch.float32)
        return self._directions

    def forward(self, data):
        coords = data[self.source]
        thresholds = torch.linspace(-self.radius, self.radius, self.resolution)
        heights = coords.float() @ self.directions(coords.shape[1])

        max_rank = len(data.triangulation[0]) - 1
        if self.max_rank is not None:
            max_rank = min(max_rank, self.max_rank)
        ranks = []
        for rank in range(max_rank + 1):
            faces = {
                face
                for simplex in data.triangulation
                for face in combinations(simplex, rank + 1)
            }
            index = torch.tensor(sorted(faces)) - 1
            face_heights = heights[index].amax(dim=1)
            # Index of the first threshold at or above each face; faces
            # above the last threshold land in a bin that is dropped.
            first = torch.searchsorted(thresholds, face_heights.T.contiguous())
            counts = torch.zeros(
                self.num_directions, self.resolution + 1, dtype=torch.long
            )
            counts.scatter_add_(1, first, torch.ones_like(first))
            ranks.append(counts[:, :-1].cumsum(dim=1))

        ranks = torch.stack(ranks)
        if ranks.max() > torch.iinfo(torch.int16).max:
            raise ValueError(
                f"{int(ranks.max())} faces of one rank do not fit int16"
            )
        if self.per_rank:
            ect = ranks
        else:
            signs = (-1) ** torch.arange(len(ranks))
            ect = (signs[:, None, None] * ranks).sum(dim=0)
        data[self.target] = ect.to(torch.int16).unsqueeze(0)
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_directions={self.num_directions}, "
            f"resolution={self.resolution}, radius={self.radius}, "
            f"max_rank={self.max_rank}, per_rank={self.per_rank}, "
            f"source={self.source!r}, target={self.target!r}, "
            f"seed={self.seed})"
        )
