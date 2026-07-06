"""Tests for ``mantra.augmentations.triangulation_3d.Triangulation3D``."""

import random

from mantra.augmentations.triangulation import Triangulation

# A single tetrahedron.
SINGLE_TET = [[1, 2, 3, 4]]
# Two tetrahedra sharing the triangular face {1, 2, 3}.
TWO_TETS_SHARED_FACE = [[1, 2, 3, 4], [1, 2, 3, 5]]
# Three tetrahedra sharing the edge {4, 5} (the result of a 2-3 move on
# TWO_TETS_SHARED_FACE).
THREE_TETS_SHARED_EDGE = [[1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]]
# Four tetrahedra around an interior vertex 5 (a subdivided tetrahedron
# {1, 2, 3, 4}); the input to a 4-1 move.
FOUR_TETS_SHARED_VERTEX = [
    [1, 2, 3, 5],
    [1, 2, 4, 5],
    [1, 3, 4, 5],
    [2, 3, 4, 5],
]


def fsets(simplices):
    return {frozenset(s) for s in simplices}


class TestMove14:
    def test_explicit_subdivides_into_four(self):
        t = Triangulation.from_list(SINGLE_TET)
        assert t.move_1_4(frozenset({1, 2, 3, 4})) is True
        assert len(t._simplices) == 4
        assert t.n_vertices == 5

    def test_random(self):
        t = Triangulation.from_list(SINGLE_TET, rng=random.Random(0))
        assert t.move_1_4() is True


class TestMove23:
    def test_explicit_success(self):
        t = Triangulation.from_list(TWO_TETS_SHARED_FACE)
        assert t.move_2_3(frozenset({1, 2, 3})) is True
        assert t._simplices == fsets(THREE_TETS_SHARED_EDGE)

    def test_random_success(self):
        t = Triangulation.from_list(TWO_TETS_SHARED_FACE, rng=random.Random(0))
        assert t.move_2_3() is True

    def test_random_no_valid_face_single_tet(self):
        # Every face has a single coface -> no valid 2-3 move.
        t = Triangulation.from_list(SINGLE_TET)
        assert t.move_2_3() is False

    def test_random_no_valid_face_when_opposite_edge_exists(self):
        # In THREE_TETS_SHARED_EDGE every internal face's opposite
        # edge already exists, so no face is a valid 2-3 candidate.
        t = Triangulation.from_list(THREE_TETS_SHARED_EDGE)
        assert t.move_2_3() is False

    def test_explicit_boundary_face_returns_false(self):
        t = Triangulation.from_list(SINGLE_TET)
        assert t.move_2_3(frozenset({1, 2, 3})) is False

    def test_explicit_blocked_when_opposite_edge_exists(self):
        # Face {1, 4, 5}: opposite vertices are 2 and 3, and edge
        # {2, 3} already exists in tet {2, 3, 4, 5}.
        t = Triangulation.from_list(THREE_TETS_SHARED_EDGE)
        assert t.move_2_3(frozenset({1, 4, 5})) is False


class TestMove32:
    def test_explicit_success(self):
        t = Triangulation.from_list(THREE_TETS_SHARED_EDGE)
        assert t.move_3_2(frozenset({4, 5})) is True
        assert t._simplices == fsets(TWO_TETS_SHARED_FACE)

    def test_random_success(self):
        t = Triangulation.from_list(THREE_TETS_SHARED_EDGE, rng=random.Random(0))
        assert t.move_3_2() is True

    def test_random_no_candidate_single_tet(self):
        t = Triangulation.from_list(SINGLE_TET)
        assert t.move_3_2() is False

    def test_explicit_wrong_number_of_containing_tets(self):
        # Edge {1, 2} is in only one tetrahedron.
        t = Triangulation.from_list(SINGLE_TET)
        assert t.move_3_2(frozenset({1, 2})) is False

    def test_explicit_link_not_a_triangle(self):
        # Three tets share edge {5, 6} but their link has four vertices,
        # so the edge is not a valid 3-2 candidate.
        cfg = [[1, 2, 5, 6], [1, 3, 5, 6], [1, 4, 5, 6]]
        t = Triangulation.from_list(cfg)
        assert t.move_3_2(frozenset({5, 6})) is False

    def test_explicit_new_face_used_elsewhere(self):
        # The new face {1, 2, 3} would collide with an existing
        # tetrahedron {1, 2, 3, 9}, so the move is rejected.
        cfg = THREE_TETS_SHARED_EDGE + [[1, 2, 3, 9]]
        t = Triangulation.from_list(cfg)
        assert t.move_3_2(frozenset({4, 5})) is False


class TestMove41:
    def test_explicit_success(self):
        t = Triangulation.from_list(FOUR_TETS_SHARED_VERTEX)
        assert t.move_4_1(5) is True
        assert t._simplices == {frozenset({1, 2, 3, 4})}

    def test_random_success(self):
        t = Triangulation.from_list(FOUR_TETS_SHARED_VERTEX, rng=random.Random(0))
        assert t.move_4_1() is True

    def test_random_no_candidate_single_tet(self):
        t = Triangulation.from_list(SINGLE_TET)
        assert t.move_4_1() is False

    def test_explicit_wrong_star_size(self):
        # Vertex 1 of a single tet has a star of one tetrahedron.
        t = Triangulation.from_list(SINGLE_TET)
        assert t.move_4_1(1) is False

    def test_explicit_link_too_large(self):
        # Vertex 5 sits in four tets but the link has five vertices.
        cfg = [[1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5], [1, 2, 6, 5]]
        t = Triangulation.from_list(cfg)
        assert t.move_4_1(5) is False

    def test_explicit_new_tet_already_exists(self):
        # The collapsed tet {1, 2, 3, 4} already exists in the complex.
        cfg = FOUR_TETS_SHARED_VERTEX + [[1, 2, 3, 4]]
        t = Triangulation.from_list(cfg)
        assert t.move_4_1(5) is False


class TestFindCandidatesViaRandom:
    def test_3_2_skips_edge_with_large_link(self):
        # Drives _find_3_2_candidates through the count!=3 and
        # link_verts!=3 continues; no candidate -> False.
        cfg = [[1, 2, 5, 6], [1, 3, 5, 6], [1, 4, 5, 6]]
        t = Triangulation.from_list(cfg, rng=random.Random(0))
        assert t.move_3_2() is False

    def test_3_2_skips_when_new_face_used_elsewhere(self):
        cfg = THREE_TETS_SHARED_EDGE + [[1, 2, 3, 9]]
        t = Triangulation.from_list(cfg, rng=random.Random(0))
        assert t.move_3_2() is False

    def test_4_1_skips_vertex_with_large_link(self):
        cfg = [[1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5], [1, 2, 6, 5]]
        t = Triangulation.from_list(cfg, rng=random.Random(0))
        assert t.move_4_1() is False

    def test_4_1_skips_when_new_tet_exists(self):
        cfg = FOUR_TETS_SHARED_VERTEX + [[1, 2, 3, 4]]
        t = Triangulation.from_list(cfg, rng=random.Random(0))
        assert t.move_4_1() is False


class TestRandomPachnerMove:
    def test_default_weights_always_succeeds(self):
        # move_1_4 always succeeds, so a random move is guaranteed.
        t = Triangulation.from_list(SINGLE_TET, rng=random.Random(0))
        assert t.random_pachner_move() is True

    def test_explicit_weights(self):
        t = Triangulation.from_list(SINGLE_TET, rng=random.Random(1))
        assert t.random_pachner_move(weights=(1.0, 1.0, 1.0, 1.0)) is True
