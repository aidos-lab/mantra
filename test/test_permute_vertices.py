"""Tests for ``mantra.transforms.PermuteVertices``."""

from collections import Counter

import pytest
from torch_geometric.data import Data

from mantra.representations.internal import SimplexTrie
from mantra.transforms import PermuteVertices

# Octahedron with antipodal pairs (1, 2), (3, 4) and (5, 6).
OCTAHEDRON_TRI = [
    [1, 3, 5],
    [1, 3, 6],
    [1, 4, 5],
    [1, 4, 6],
    [2, 3, 5],
    [2, 3, 6],
    [2, 4, 5],
    [2, 4, 6],
]


def _degrees(triangulation):
    return sorted(Counter(v for s in triangulation for v in s).values())


def test_requires_triangulation():
    with pytest.raises(AssertionError):
        PermuteVertices(rng=0)(Data(n_vertices=6))


def test_rejects_zero_indexed_labels():
    zero_indexed = [[v - 1 for v in s] for s in OCTAHEDRON_TRI]
    with pytest.raises(ValueError):
        PermuteVertices(rng=0)(Data(triangulation=zero_indexed))


def test_rejects_label_gaps():
    with pytest.raises(ValueError):
        PermuteVertices(rng=0)(Data(triangulation=[[1, 2, 3], [2, 3, 5]]))


def test_rejects_stale_simplex_trie():
    data = Data(triangulation=OCTAHEDRON_TRI, simplex_trie=SimplexTrie())
    with pytest.raises(ValueError):
        PermuteVertices(rng=0)(data)


def test_relabels_triangulation():
    # A draw can be one of the 48 automorphisms, so try a few seeds.
    results = [
        PermuteVertices(rng=seed)(
            Data(triangulation=OCTAHEDRON_TRI, n_vertices=6)
        ).triangulation
        for seed in range(5)
    ]

    assert any(r != OCTAHEDRON_TRI for r in results)


def test_keeps_labels_and_combinatorics():
    data = Data(triangulation=OCTAHEDRON_TRI, n_vertices=6)
    result = PermuteVertices(rng=0)(data)

    assert {v for s in result.triangulation for v in s} == set(range(1, 7))
    assert len(result.triangulation) == len(OCTAHEDRON_TRI)
    assert _degrees(result.triangulation) == _degrees(OCTAHEDRON_TRI)


def test_output_is_sorted():
    data = Data(triangulation=OCTAHEDRON_TRI, n_vertices=6)
    result = PermuteVertices(rng=3)(data)

    assert all(s == sorted(s) for s in result.triangulation)
    assert result.triangulation == sorted(result.triangulation)


def test_is_reproducible_with_seed():
    data_a = Data(triangulation=OCTAHEDRON_TRI, n_vertices=6)
    data_b = Data(triangulation=OCTAHEDRON_TRI, n_vertices=6)

    result_a = PermuteVertices(rng=11)(data_a)
    result_b = PermuteVertices(rng=11)(data_b)

    assert result_a.triangulation == result_b.triangulation


def test_does_not_touch_other_attributes():
    data = Data(triangulation=OCTAHEDRON_TRI, n_vertices=6, genus=0)
    result = PermuteVertices(rng=0)(data)

    assert result.n_vertices == 6
    assert result.genus == 0
