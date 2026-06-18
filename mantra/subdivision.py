"""Functional subdivision API for raw triangulations.

Barycentric and stellar subdivisions deterministically *refine* a fixed
triangulation (raising the vertex/simplex counts) -- a distinct operation from
the Pachner-move balancing in :mod:`mantra.augmentations`, which augments a
dataset while preserving homeomorphism type.

These functions are thin wrappers over the subdivision *moves* that live on
:class:`mantra.augmentations.base.Triangulation`. They accept and return raw
``list[list[int]]`` triangulations (1-indexed vertices, matching the LEX
convention used throughout MANTRA); the returned triangulation is canonicalised
by :meth:`Triangulation.to_list` (contiguous ``1..n`` labels, sorted simplices).
"""

from mantra.augmentations.base import Triangulation

__all__ = [
    "barycentric_subdivision_raw",
    "stellar_subdivision_raw",
    "barycentric_stellar_graded",
]


def _triangulation(triangulation, rng=None):
    """Build a Triangulation from a raw simplex list (dimension inferred).

    Raises
    ------
    ValueError
        If the triangulation is not pure (its top simplices do not all
        share the same number of vertices), since the inferred dimension
        would then be meaningless.
    """
    sizes = {len(s) for s in triangulation}
    if len(sizes) != 1:
        raise ValueError(
            f"Triangulation must be pure (all top simplices the same size); "
            f"got simplex sizes {sorted(sizes)}."
        )
    dimension = len(triangulation[0]) - 1
    return Triangulation(triangulation, dimension=dimension, rng=rng)


def barycentric_subdivision_raw(triangulation):
    """Apply barycentric subdivision to a raw triangulation.

    Parameters
    ----------
    triangulation : list of list of int
        Top-dimensional simplices (1-indexed vertices).

    Returns
    -------
    new_triangulation : list of list of int
        Subdivided top-dimensional simplices (1-indexed vertices).
    n_vertices : int
        Number of vertices in the subdivided triangulation.
    """
    if not triangulation:
        return [], 0
    t = _triangulation(triangulation)
    t.barycentric_subdivision()
    return t.to_list(), t.n_vertices


def stellar_subdivision_raw(triangulation, fraction=1.0, rng=None):
    """Stellar subdivision at a (random) fraction of top-dim simplices.

    Parameters
    ----------
    triangulation : list of list of int
        Top-dimensional simplices, 1-indexed vertices.
    fraction : float
        Fraction of top-dim simplices to subdivide, in [0, 1]. 1.0
        (default) reproduces full stellar subdivision.
    rng : random.Random or None
        Source of randomness for selecting the subset. Ignored when
        fraction == 1.0.

    Returns
    -------
    new_triangulation : list of list of int
    n_vertices : int
    """
    if not triangulation:
        return [], 0
    t = _triangulation(triangulation, rng=rng)
    t.stellar_subdivision(fraction)
    return t.to_list(), t.n_vertices


def barycentric_stellar_graded(triangulation, over_vrtx_cnt, rng=None):
    """Subdivide the triangulation until there are ``over_vrtx_cnt`` vertices.

    Parameters
    ----------
    triangulation : list of list of int
        Top-dimensional simplices, 1-indexed vertices.
    over_vrtx_cnt : int
        Target vertex count to grow the triangulation to. Must be strictly
        greater than the current vertex count.
    rng : random.Random or None
        Source of randomness for choosing simplices to subdivide.

    Returns
    -------
    new_triangulation : list of list of int
    n_vertices : int

    Raises
    ------
    ValueError
        If ``over_vrtx_cnt`` is not strictly above the current vertex
        count (graded subdivision only adds vertices).
    """
    if not triangulation:
        return [], 0
    t = _triangulation(triangulation, rng=rng)
    t.graded_subdivision(over_vrtx_cnt)
    return t.to_list(), t.n_vertices
