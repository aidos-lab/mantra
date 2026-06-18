"""Tests for the subdivision moves on the Triangulation classes."""

import random
from itertools import combinations as combs

import pytest

from mantra.augmentations.base import Triangulation
from mantra.augmentations.triangulation_2d import Triangulation2D
from mantra.augmentations.triangulation_3d import Triangulation3D


def _euler(triangulation):
    """Euler characteristic of a raw triangulation (any dimension)."""
    cells = {}
    for simplex in triangulation:
        s = tuple(sorted(simplex))
        for k in range(1, len(s) + 1):
            for face in combs(s, k):
                cells.setdefault(k - 1, set()).add(face)
    return sum((-1) ** k * len(faces) for k, faces in cells.items())


class TestStellarSubdivide:
    """Single-simplex stellar move (1-(d+1) Pachner move)."""

    def test_triangle(self):
        t = Triangulation([[1, 2, 3]], dimension=2)
        t.stellar_subdivide()
        assert t.n_vertices == 4
        assert len(t.to_list()) == 3

    def test_tetrahedron(self):
        t = Triangulation([[1, 2, 3, 4]], dimension=3)
        t.stellar_subdivide()
        assert t.n_vertices == 5
        assert len(t.to_list()) == 4

    def test_explicit_simplex(self):
        t = Triangulation([[1, 2, 3], [2, 3, 4]], dimension=2)
        t.stellar_subdivide(frozenset({1, 2, 3}))
        # one triangle became three, the other is untouched
        assert len(t.to_list()) == 4
        assert t.n_vertices == 5


class TestStellarSubdivision:
    """Aggregate stellar subdivision over a fraction of simplices."""

    SPHERE = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]

    def test_full(self):
        t = Triangulation2D(self.SPHERE)
        t.stellar_subdivision(1.0)
        # 4 triangles * 3 + 4 barycenters
        assert len(t.to_list()) == 12
        assert t.n_vertices == 8

    def test_partial_with_rng(self):
        t = Triangulation2D(self.SPHERE, rng=random.Random(0))
        t.stellar_subdivision(0.5)
        # 2 subdivided (3 each) + 2 passed through
        assert len(t.to_list()) == 8
        assert t.n_vertices == 6

    def test_fraction_out_of_range(self):
        t = Triangulation2D(self.SPHERE)
        with pytest.raises(ValueError):
            t.stellar_subdivision(1.5)

    def test_small_fraction_subdivides_at_least_one(self):
        # round(0.1 * 4) == 0; a positive fraction must still subdivide one
        # simplex rather than silently passing the complex through unchanged.
        t = Triangulation2D(self.SPHERE, rng=random.Random(0))
        t.stellar_subdivision(0.1)
        assert t.n_vertices == 5  # 4 original + 1 barycenter
        assert len(t.to_list()) == 6  # one triangle -> 3, plus 3 untouched

    def test_zero_fraction_is_noop(self):
        t = Triangulation2D(self.SPHERE)
        t.stellar_subdivision(0.0)
        assert t.n_vertices == 4
        assert len(t.to_list()) == 4


class TestBarycentricSubdivision:
    """Full barycentric (order-complex) subdivision."""

    def test_triangle(self):
        t = Triangulation([[1, 2, 3]], dimension=2)
        t.barycentric_subdivision()
        assert t.n_vertices == 7
        assert len(t.to_list()) == 6

    def test_tetrahedron(self):
        t = Triangulation([[1, 2, 3, 4]], dimension=3)
        t.barycentric_subdivision()
        assert t.n_vertices == 15
        assert len(t.to_list()) == 24

    def test_euler_characteristic_preserved(self):
        sphere = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        t = Triangulation2D(sphere)
        assert _euler(t.to_list()) == 2
        t.barycentric_subdivision()
        assert _euler(t.to_list()) == 2


class TestGradedSubdivision:
    """Graded stellar subdivision up to a vertex target."""

    def test_target_2d(self):
        t = Triangulation([[1, 2, 3]], dimension=2)
        t.graded_subdivision(5)
        assert t.n_vertices == 5
        assert len(t.to_list()) == 5

    def test_target_3d(self):
        t = Triangulation([[1, 2, 3, 4]], dimension=3)
        t.graded_subdivision(6)
        assert t.n_vertices == 6
        assert len(t.to_list()) == 7

    def test_target_at_or_below_current_raises(self):
        # The point of graded subdivision is to refine into unseen data, so a
        # target that adds no vertices (at or below the current count) is
        # rejected rather than silently doing nothing.
        t = Triangulation([[1, 2, 3]], dimension=2)
        with pytest.raises(ValueError):
            t.graded_subdivision(3)
        with pytest.raises(ValueError):
            t.graded_subdivision(2)
        # The triangulation is left untouched.
        assert t.n_vertices == 3
        assert t.to_list() == [[1, 2, 3]]


class TestPachnerAliasesUnchanged:
    """The delegating Pachner names keep their original behaviour."""

    def test_2d_subdivide(self):
        t = Triangulation2D([[1, 2, 3]])
        assert t.subdivide() is True
        assert t.n_vertices == 4
        assert len(t.to_list()) == 3

    def test_3d_move_1_4(self):
        t = Triangulation3D([[1, 2, 3, 4]])
        assert t.move_1_4() is True
        assert t.n_vertices == 5
        assert len(t.to_list()) == 4
        # 5 is the new center vertex
        assert t.to_list() == [
            [1, 2, 3, 5],
            [1, 2, 4, 5],
            [1, 3, 4, 5],
            [2, 3, 4, 5],
        ]
