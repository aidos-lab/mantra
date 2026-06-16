"""Tests for ``mantra.augmentations.triangulation_2d.Triangulation2D``."""

import random

from mantra.augmentations.triangulation_2d import (
    _RP2_TRIANGULATION_MINUS_FACE,
    _TORUS_TRIANGULATION_MINUS_FACE,
    Triangulation2D,
)

# Two triangles sharing edge {2, 3}; the only flippable edge.
TWO_TRIANGLES = [[1, 2, 3], [2, 3, 4]]
# Boundary of a tetrahedron (a 2-sphere): no edge is flippable because
# the opposite edge always already exists.
SPHERE = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]


class TestFlipEdge:
    def test_explicit_flip(self):
        t = Triangulation2D(TWO_TRIANGLES)
        assert t.flip_edge(frozenset({2, 3})) is True
        assert t._simplices == {frozenset({1, 2, 4}), frozenset({1, 3, 4})}

    def test_random_flip_picks_the_flippable_edge(self):
        t = Triangulation2D(TWO_TRIANGLES, rng=random.Random(0))
        assert t.flip_edge() is True
        assert frozenset({1, 4}) in {
            s & frozenset({1, 4}) or s for s in t._simplices
        }

    def test_random_no_flippable_edge_single_triangle(self):
        # Every edge has a single coface -> nothing to flip.
        t = Triangulation2D([[1, 2, 3]])
        assert t.flip_edge() is False

    def test_random_no_flippable_edge_sphere(self):
        # Each edge's opposite edge already exists -> not flippable.
        t = Triangulation2D(SPHERE)
        assert t.flip_edge() is False

    def test_explicit_boundary_edge_returns_false(self):
        # Edge with only one coface cannot be flipped.
        t = Triangulation2D([[1, 2, 3]])
        assert t.flip_edge(frozenset({1, 2})) is False

    def test_explicit_flip_blocked_when_new_edge_exists(self):
        # On the sphere, flipping {1, 2} would create edge {3, 4},
        # which already exists.
        t = Triangulation2D(SPHERE)
        assert t.flip_edge(frozenset({1, 2})) is False


class TestSubdivide:
    def test_explicit_subdivide_makes_three_triangles(self):
        t = Triangulation2D([[1, 2, 3]])
        assert t.subdivide(frozenset({1, 2, 3})) is True
        # Old triangle gone, three new ones around the new vertex 4.
        assert t._simplices == {
            frozenset({1, 2, 4}),
            frozenset({1, 3, 4}),
            frozenset({2, 3, 4}),
        }

    def test_random_subdivide(self):
        t = Triangulation2D([[1, 2, 3]], rng=random.Random(0))
        assert t.subdivide() is True
        assert len(t._simplices) == 3


class TestGlueTorus:
    def test_explicit_increases_genus(self):
        t = Triangulation2D(SPHERE)
        before = len(t._simplices)
        assert t.glue_torus(frozenset({1, 2, 3})) == {}
        # Removed one triangle, added the torus-minus-face piece.
        assert len(t._simplices) == before - 1 + len(
            _TORUS_TRIANGULATION_MINUS_FACE
        )

    def test_random_torus(self):
        t = Triangulation2D(SPHERE, rng=random.Random(1))
        assert t.glue_torus() == {}


class TestGlueCrosscap:
    def test_explicit_adds_crosscap(self):
        t = Triangulation2D(SPHERE)
        before = len(t._simplices)
        assert t.glue_crosscap(frozenset({1, 2, 3})) == {}
        assert len(t._simplices) == before - 1 + len(
            _RP2_TRIANGULATION_MINUS_FACE
        )

    def test_random_crosscap(self):
        t = Triangulation2D(SPHERE, rng=random.Random(1))
        assert t.glue_crosscap() == {}


class TestRandomPachnerMove:
    def test_default_weights(self):
        t = Triangulation2D(TWO_TRIANGLES, rng=random.Random(0))
        assert isinstance(t.random_pachner_move(), bool)

    def test_explicit_weights_force_subdivide(self):
        # Zero weight on flip -> subdivide is chosen, always succeeds.
        t = Triangulation2D([[1, 2, 3]], rng=random.Random(0))
        assert t.random_pachner_move(weights=(0.0, 1.0)) is True
