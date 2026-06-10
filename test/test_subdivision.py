"""Unit tests for :mod:`mantra.subdivision`."""

import random

import pytest

from mantra.subdivision import (
    barycentric_subdivision_raw,
    barycentric_stellar_graded,
    stellar_subdivision_raw,
    subdivide_dataset_json,
)


def test_barycentric_single_triangle():
    # A 2-simplex has 3 vertices + 3 edges + 1 face = 7 sub-simplices,
    # and 3! = 6 flags (top-dimensional simplices) in its subdivision.
    tri, n_v = barycentric_subdivision_raw([[1, 2, 3]])
    assert n_v == 7
    assert len(tri) == 6
    assert all(len(s) == 3 for s in tri)
    # Every new vertex label is used and within range.
    used = {v for s in tri for v in s}
    assert used == set(range(1, 8))


def test_stellar_full_single_triangle():
    # Full stellar subdivision of one triangle: 1 barycenter, 3 sub-triangles.
    tri, n_v = stellar_subdivision_raw([[1, 2, 3]], fraction=1.0)
    assert n_v == 4
    assert len(tri) == 3
    assert all(4 in s for s in tri)


def test_stellar_fraction_zero_is_identity():
    original = [[1, 2, 3], [2, 3, 4]]
    tri, n_v = stellar_subdivision_raw(original, fraction=0.0)
    assert n_v == 4
    assert sorted(sorted(s) for s in tri) == sorted(sorted(s) for s in original)


def test_stellar_invalid_fraction_raises():
    with pytest.raises(ValueError):
        stellar_subdivision_raw([[1, 2, 3]], fraction=1.5)


def test_stellar_empty_triangulation():
    assert stellar_subdivision_raw([]) == ([], 0)


def test_barycentric_empty_triangulation():
    assert barycentric_subdivision_raw([]) == ([], 0)


def test_stellar_seeded_determinism():
    original = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
    a = stellar_subdivision_raw(original, fraction=0.5, rng=random.Random(0))
    b = stellar_subdivision_raw(original, fraction=0.5, rng=random.Random(0))
    assert a == b


def test_subdivide_dataset_json_preserves_metadata():
    data = [
        {
            "triangulation": [[1, 2, 3]],
            "n_vertices": 3,
            "name": "S^2",
            "betti_numbers": [1, 0, 1],
            "dimension": 2,
        }
    ]
    out = subdivide_dataset_json(data, n_subdivisions=1)
    assert len(out) == 1
    entry = out[0]
    assert entry["name"] == "S^2"
    assert entry["betti_numbers"] == [1, 0, 1]
    assert entry["dimension"] == 2
    assert entry["n_vertices"] == 7  # one barycentric round on a triangle
    # Original input is not mutated.
    assert data[0]["n_vertices"] == 3


def test_graded_reaches_vertex_target():
    tri, n_v = barycentric_stellar_graded(
        [[1, 2, 3], [2, 3, 4]], over_vrtx_cnt=6, rng=random.Random(0)
    )
    assert n_v == 6
    assert len(tri) > 0
