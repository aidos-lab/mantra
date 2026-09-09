"""Tests for ``mantra.augmentations.triangulation_2d.Triangulation2D``."""

import random

from mantra.utils.constants import (
    RP2_TRIANGULATION_MINUS_FACE,
    TORUS_TRIANGULATION_MINUS_FACE,
)
from mantra.utils.triangulation import Triangulation, Triangulation2D

# Two triangles sharing edge {2, 3}; the only flippable edge.
TWO_TRIANGLES = [[1, 2, 3], [2, 3, 4]]
# Boundary of a tetrahedron (a 2-sphere): no edge is flippable because
# the opposite edge always already exists.
SPHERE = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]


class TestFlipEdge:
    def test_explicit_flip(self):
        t = Triangulation.from_list(TWO_TRIANGLES)
        assert t.flip_edge(frozenset({2, 3})) is True
        assert t._simplices == {frozenset({1, 2, 4}), frozenset({1, 3, 4})}

    def test_random_flip_picks_the_flippable_edge(self):
        t = Triangulation.from_list(TWO_TRIANGLES, rng=random.Random(0))
        assert t.flip_edge() is True
        assert frozenset({1, 4}) in {
            s & frozenset({1, 4}) or s for s in t._simplices
        }

    def test_random_no_flippable_edge_single_triangle(self):
        # Every edge has a single coface -> nothing to flip.
        t = Triangulation.from_list([[1, 2, 3]])
        assert t.flip_edge() is False

    def test_random_no_flippable_edge_sphere(self):
        # Each edge's opposite edge already exists -> not flippable.
        t = Triangulation.from_list(SPHERE)
        assert t.flip_edge() is False

    def test_explicit_boundary_edge_returns_false(self):
        # Edge with only one coface cannot be flipped.
        t = Triangulation.from_list([[1, 2, 3]])
        assert t.flip_edge(frozenset({1, 2})) is False

    def test_explicit_flip_blocked_when_new_edge_exists(self):
        # On the sphere, flipping {1, 2} would create edge {3, 4},
        # which already exists.
        t = Triangulation.from_list(SPHERE)
        assert t.flip_edge(frozenset({1, 2})) is False


class TestSubdivide:
    def test_explicit_subdivide_makes_three_triangles(self):
        t = Triangulation.from_list([[1, 2, 3]])
        assert t.subdivide(frozenset({1, 2, 3})) is True
        # Old triangle gone, three new ones around the new vertex 4.
        assert t._simplices == {
            frozenset({1, 2, 4}),
            frozenset({1, 3, 4}),
            frozenset({2, 3, 4}),
        }

    def test_random_subdivide(self):
        t = Triangulation.from_list([[1, 2, 3]], rng=random.Random(0))
        assert t.subdivide() is True
        assert len(t._simplices) == 3


class TestGlueTorus:
    def test_explicit_increases_genus(self):
        t = Triangulation.from_list(SPHERE)
        before = len(t._simplices)
        assert t.glue("torus", frozenset({1, 2, 3})) == {}
        # Removed one triangle, added the torus-minus-face piece.
        assert len(t._simplices) == before - 1 + len(
            TORUS_TRIANGULATION_MINUS_FACE
        )

    def test_random_torus(self):
        t = Triangulation.from_list(SPHERE, rng=random.Random(1))
        assert t.glue("torus") == {}


class TestGlueCrosscap:
    def test_explicit_adds_crosscap(self):
        t = Triangulation.from_list(SPHERE)
        before = len(t._simplices)
        assert t.glue("crosscap", frozenset({1, 2, 3})) == {}
        assert len(t._simplices) == before - 1 + len(
            RP2_TRIANGULATION_MINUS_FACE
        )

    def test_random_crosscap(self):
        t = Triangulation.from_list(SPHERE, rng=random.Random(1))
        assert t.glue("crosscap") == {}


class TestRandomPachnerMove:
    def test_default_weights(self):
        t = Triangulation.from_list(TWO_TRIANGLES, rng=random.Random(0))
        assert isinstance(t.random_pachner_move(), bool)

    def test_explicit_weights_force_subdivide(self):
        # Zero weight on flip -> subdivide is chosen, always succeeds.
        t = Triangulation.from_list([[1, 2, 3]], rng=random.Random(0))
        assert t.random_pachner_move(weights=(0.0, 1.0, 0.0)) is True

    def test_zero_weights_are_never_tried(self):
        # Only the 3-1 move is allowed and the sphere has no removable
        # vertex, so no move is possible.
        t = Triangulation.from_list(SPHERE, rng=random.Random(0))
        assert t.random_pachner_move(weights=(0.0, 0.0, 1.0)) is False

    def test_falls_back_to_a_possible_move(self):
        # Flip has almost all the weight but is impossible on the
        # sphere; the move must fall back to the subdivision.
        t = Triangulation.from_list(SPHERE, rng=random.Random(0))
        assert t.random_pachner_move(weights=(1e6, 1.0, 0.0)) is True
        assert t.n_vertices == 5

    def test_remove_only_weights_only_remove_vertices(self):
        t = Triangulation.from_list(SPHERE, rng=random.Random(0))
        for _ in range(4):
            t.subdivide()
        assert t.n_vertices == 8
        for expected in (7, 6, 5, 4):
            assert t.random_pachner_move(weights=(0.0, 0.0, 1.0)) is True
            assert t.n_vertices == expected


class TestMove31:
    def test_explicit_removal_undoes_subdivision(self):
        t = Triangulation.from_list(SPHERE)
        before = set(t._simplices)
        assert t.subdivide(frozenset({1, 2, 3})) is True
        assert t.move_3_1(5) is True
        assert t._simplices == before

    def test_random_removal_gives_back_a_tetrahedral_sphere(self):
        # After subdividing {1, 2, 3} both the new vertex 5 and the
        # opposite vertex 4 are removable (the link of 4 is no longer
        # a face); either choice yields the boundary of a tetrahedron.
        t = Triangulation.from_list(SPHERE, rng=random.Random(0))
        t.subdivide(frozenset({1, 2, 3}))
        assert t.move_3_1() is True
        assert t.n_vertices == 4
        assert len(t._simplices) == 4
        t.validate()

    def test_no_candidate_on_sphere(self):
        # Every vertex of the tetrahedral sphere has degree 3, but its
        # link triangle is already a face of the complex.
        t = Triangulation.from_list(SPHERE, rng=random.Random(0))
        assert t.move_3_1() is False
        assert t.move_3_1(1) is False

    def test_rejects_vertex_of_wrong_degree(self):
        t = Triangulation.from_list(SPHERE)
        t.subdivide(frozenset({1, 2, 3}))
        t.subdivide(frozenset({1, 2, 5}))
        # Vertex 5 now has degree 4.
        assert t.move_3_1(5) is False
        assert t.n_vertices == 6

    def test_rejects_link_that_is_not_a_triangle(self):
        # Vertex 1 has degree 3, but its link is a path, not a cycle.
        t = Triangulation.from_list([[1, 2, 3], [1, 3, 4], [1, 4, 5]])
        assert t.move_3_1(1) is False
        assert len(t._simplices) == 3

    def test_aliases(self):
        assert Triangulation2D.move_1_3 is Triangulation2D.subdivide
        assert Triangulation2D.move_2_2 is Triangulation2D.flip_edge
