"""Tests for ``mantra.representations.internal.Simplex``."""

import pytest

from mantra.representations.internal import Simplex


def test_iterates_in_ascending_order():
    # A frozenset would iterate {1, 8} as (8, 1).
    assert tuple(Simplex((8, 1))) == (1, 8)
    assert repr(Simplex((10, 3))) == "Simplex((3, 10))"


def test_sorts_lexicographically():
    simplices = [
        Simplex((1, 8)),
        Simplex((1, 2)),
        Simplex((2, 3)),
        Simplex((1, 10)),
    ]

    assert [tuple(s) for s in sorted(simplices)] == [
        (1, 2),
        (1, 8),
        (1, 10),
        (2, 3),
    ]


def test_equality_and_hash_ignore_input_order():
    assert Simplex((8, 1)) == Simplex((1, 8))
    assert hash(Simplex((8, 1))) == hash(Simplex((1, 8)))
    assert len({Simplex((8, 1)), Simplex((1, 8))}) == 1


def test_rejects_duplicates():
    with pytest.raises(ValueError):
        Simplex((1, 1, 2))


def test_contains_vertex():
    simplex = Simplex((1, 8))

    assert 8 in simplex
    assert 2 not in simplex
