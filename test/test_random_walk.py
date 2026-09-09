"""Tests for the move log and ``Triangulation.random_walk``."""

import random
from itertools import combinations

import pytest

from mantra.utils.constants import TORUS_TRIANGULATION_MINUS_FACE
from mantra.utils.triangulation import Triangulation

SPHERE = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
TORUS = TORUS_TRIANGULATION_MINUS_FACE + [[1, 2, 3]]
S3 = [list(c) for c in combinations(range(1, 6), 4)]


class TestMoveLog:
    def test_starts_empty(self):
        assert Triangulation.from_list(SPHERE).move_log == []

    def test_2d_moves_are_logged_with_their_simplex(self):
        t = Triangulation.from_list(TORUS)
        assert t.subdivide(frozenset({4, 1, 2})) is True
        assert t.flip_edge(frozenset({2, 1})) is True
        assert t.flip_edge(frozenset({3, 8})) is True
        assert t.move_3_1(8) is True
        assert t.move_log == [
            ("1-3", (1, 2, 4)),
            ("2-2", (1, 2)),
            ("2-2", (3, 8)),
            ("3-1", (8,)),
        ]

    def test_3d_moves_are_logged_with_their_simplex(self):
        t = Triangulation.from_list(S3)
        assert t.move_1_4(frozenset({4, 3, 2, 1})) is True
        assert t.move_2_3(frozenset({3, 2, 1})) is True
        assert t.move_3_2(frozenset({6, 5})) is True
        assert t.move_4_1(6) is True
        assert t.move_log == [
            ("1-4", (1, 2, 3, 4)),
            ("2-3", (1, 2, 3)),
            ("3-2", (5, 6)),
            ("4-1", (6,)),
        ]

    def test_failed_moves_are_not_logged(self):
        t = Triangulation.from_list(SPHERE)
        assert t.flip_edge(frozenset({1, 2})) is False
        assert t.move_3_1(1) is False
        assert t.move_log == []


class TestRandomWalk:
    @pytest.mark.parametrize("start, chi", [(TORUS, 0), (S3, 0)])
    def test_snapshots_validate_and_keep_chi(self, start, chi):
        t = Triangulation.from_list(start, rng=random.Random(3))
        snapshots = t.random_walk(n_steps=5, moves_per_step=3)
        assert len(snapshots) == 6
        assert snapshots[0] == Triangulation.from_list(start).to_list()
        assert snapshots[-1] == t.to_list()
        assert len(t.move_log) == 15
        for tri in snapshots:
            s = Triangulation.from_list(tri)
            s.validate()
            assert s.euler_characteristic() == chi

    def test_zero_steps_returns_only_the_start(self):
        t = Triangulation.from_list(TORUS, rng=random.Random(3))
        assert t.random_walk(n_steps=0) == [t.to_list()]
        assert t.move_log == []

    def test_is_deterministic_for_a_seed(self):
        runs = []
        for _ in range(2):
            t = Triangulation.from_list(TORUS, rng=random.Random(11))
            runs.append(
                (t.random_walk(n_steps=4, moves_per_step=2), t.move_log)
            )
        assert runs[0] == runs[1]

    def test_different_seeds_differ(self):
        t1 = Triangulation.from_list(TORUS, rng=random.Random(1))
        t2 = Triangulation.from_list(TORUS, rng=random.Random(2))
        assert t1.random_walk(n_steps=4) != t2.random_walk(n_steps=4)

    def test_raises_when_no_move_is_possible(self):
        t = Triangulation.from_list(SPHERE, rng=random.Random(0))
        with pytest.raises(RuntimeError, match="No Pachner move"):
            t.random_walk(n_steps=1, weights=(1.0, 0.0, 1.0))
