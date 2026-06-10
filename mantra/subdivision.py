"""Pure subdivision operations for raw triangulations.

Barycentric and stellar subdivisions deterministically *refine* a fixed
triangulation (raising the vertex/simplex counts) — a distinct operation from
the Pachner-move balancing in :mod:`mantra.augmentations`, which augments a
dataset while preserving homeomorphism type.

These functions are pure Python (only the standard library) and operate on raw
``list[list[int]]`` triangulations (1-indexed vertices, matching the LEX
convention used throughout MANTRA). They have no ML dependencies and are safe
to import anywhere.
"""

import random
from itertools import combinations, permutations


__all__ = [
    "barycentric_subdivision_raw",
    "stellar_subdivision_raw",
    "barycentric_stellar_graded",
    "subdivide_dataset_json",
]


def barycentric_subdivision_raw(triangulation):
    """Apply barycentric subdivision to a raw triangulation.

    Each sub-simplex of each top-dimensional simplex becomes a new vertex.
    Each permutation of vertices in a top-dimensional simplex produces
    a new top-dimensional simplex in the subdivision.

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
    all_simplices = {}
    next_vertex = 1

    for simplex in triangulation:
        simplex_sorted = sorted(simplex)
        for k in range(1, len(simplex_sorted) + 1):
            for face in combinations(simplex_sorted, k):
                key = frozenset(face)
                if key not in all_simplices:
                    all_simplices[key] = next_vertex
                    next_vertex += 1

    new_triangulation = []
    for simplex in triangulation:
        simplex_sorted = sorted(simplex)
        for perm in permutations(simplex_sorted):
            new_simplex = []
            for k in range(1, len(perm) + 1):
                prefix = frozenset(perm[:k])
                new_simplex.append(all_simplices[prefix])
            new_triangulation.append(new_simplex)

    return new_triangulation, next_vertex - 1


def barycentric_stellar_graded(triangulation, over_vrtx_cnt, rng=None):
    """Subdivide the triangulation until there are `over_vrtx_cnt` vertices."""

    if not triangulation:
        return [], 0

    original = set()

    # Add vertices of all top-level simplices
    for simplex in triangulation:
        original.add(frozenset(simplex))

    # Next vertex numbering
    next_v = max([max(s) for s in list(original)]) + 1

    # Init rng
    if rng is None:
        rng = random.Random()

    already_chosen = set()
    running = True

    running_simplices = set()
    while running:
        # Check available simplices
        available_for_choosing = original - already_chosen

        # In this case we need to perform further subdivision
        if len(available_for_choosing) == 0:
            original.update(running_simplices)
            running_simplices = set()
            already_chosen = set()
            continue

        # Pick a random top level simplex
        chosen = rng.choice(list(available_for_choosing))

        # Remove from other simplices
        original.remove(chosen)

        # Get all combinations of k-1 vertices of the k-simplex  (actually k+1)
        new_simplices = set(
            [
                frozenset(list(s) + [next_v])
                for s in list(combinations(chosen, len(chosen) - 1))
            ]
        )

        running_simplices.update(new_simplices)

        # next node
        next_v += 1

        # Update already chosen
        already_chosen.add(frozenset(chosen))

        # We reached the max_vrtx cnt
        if next_v == over_vrtx_cnt + 1:
            original.update(running_simplices)
            break

    return [sorted(s) for s in original], next_v - 1


def stellar_subdivision_raw(triangulation, fraction=1.0, rng=None):
    """Stellar subdivision at a (random) fraction of top-dim simplices.

    For each chosen top-dim simplex σ, add a new vertex b_σ (its
    barycenter) and replace σ with the d+1 sub-simplices
    (σ \\ {v}) ∪ {b_σ} for v ∈ σ. Skipped simplices are passed through
    unchanged. ∂σ is unchanged for both chosen and skipped simplices, so
    partial application preserves the simplicial-complex condition
    across (d-1)-faces shared with neighbours. Original vertex labels
    are preserved.

    Parameters
    ----------
    triangulation : list of list of int
        Top-dimensional simplices, 1-indexed vertices, dense labels.
    fraction : float
        Fraction of top-dim simplices to subdivide, in [0, 1]. 1.0
        (default) reproduces full stellar subdivision.
    rng : random.Random or None
        Source of randomness for selecting the subset. Ignored when
        fraction == 1.0. If None for fraction < 1.0, a fresh
        ``random.Random()`` is used.

    Returns
    -------
    new_triangulation : list of list of int
    n_vertices : int
    """
    if not triangulation:
        return [], 0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    original = set()
    for simplex in triangulation:
        original.update(simplex)
    next_v = max(original) + 1

    n_total = len(triangulation)
    if fraction == 1.0:
        chosen = None  # subdivide all; no RNG needed
    else:
        n_subdivide = round(fraction * n_total)
        if rng is None:
            rng = random.Random()
        chosen = set(rng.sample(range(n_total), n_subdivide))

    new_triangulation = []
    for idx, simplex in enumerate(triangulation):
        if chosen is not None and idx not in chosen:
            new_triangulation.append(list(simplex))
            continue
        b = next_v
        next_v += 1
        for i in range(len(simplex)):
            new_triangulation.append(
                sorted(list(simplex[:i]) + list(simplex[i + 1:]) + [b])
            )
    return new_triangulation, next_v - 1


def subdivide_dataset_json(data, n_subdivisions=1):
    """Apply n rounds of barycentric subdivision to all triangulations.

    Parameters
    ----------
    data : list of dict
        Raw dataset entries with 'triangulation' and 'n_vertices' keys.
    n_subdivisions : int
        Number of subdivision rounds.

    Returns
    -------
    list of dict
        Dataset entries with subdivided triangulations.
    """
    result = []
    for entry in data:
        tri = entry["triangulation"]
        n_v = entry["n_vertices"]
        for _ in range(n_subdivisions):
            tri, n_v = barycentric_subdivision_raw(tri)
        new_entry = dict(entry)
        new_entry["triangulation"] = tri
        new_entry["n_vertices"] = n_v
        result.append(new_entry)
    return result
