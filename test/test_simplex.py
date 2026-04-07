"""Tests for Simplex class."""

import pytest

from mantra.representations.internal.simplex import Simplex


class TestSimplexCreation:
    def test_create_vertex(self):
        s = Simplex((1,))
        assert len(s) == 1
        assert 1 in s

    def test_create_edge(self):
        s = Simplex((1, 2))
        assert len(s) == 2

    def test_create_triangle(self):
        s = Simplex((1, 2, 3))
        assert len(s) == 3

    def test_duplicate_elements_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            Simplex((1, 1, 2))


class TestSimplexEquality:
    def test_equal_same_order(self):
        assert Simplex((1, 2, 3)) == Simplex((1, 2, 3))

    def test_equal_different_order(self):
        assert Simplex((1, 2, 3)) == Simplex((3, 1, 2))

    def test_not_equal_different_elements(self):
        assert Simplex((1, 2, 3)) != Simplex((1, 2, 4))

    def test_not_equal_different_type(self):
        assert Simplex((1, 2)) != (1, 2)


class TestSimplexHash:
    def test_equal_simplices_same_hash(self):
        assert hash(Simplex((1, 2, 3))) == hash(Simplex((3, 1, 2)))

    def test_usable_in_set(self):
        s = {Simplex((1, 2)), Simplex((2, 1)), Simplex((1, 3))}
        assert len(s) == 2


class TestSimplexContains:
    def test_element_in_simplex(self):
        s = Simplex((1, 2, 3))
        assert 1 in s
        assert 4 not in s


class TestSimplexIteration:
    def test_iter(self):
        s = Simplex((1, 2, 3))
        assert set(s) == {1, 2, 3}


class TestSimplexOrdering:
    def test_le(self):
        # Ordering is based on frozenset tuples, so this is
        # nondeterministic for general types. For ints it's consistent.
        s1 = Simplex((1, 2))
        s2 = Simplex((1, 2))
        assert s1 <= s2

    def test_le_requires_simplex(self):
        with pytest.raises(AssertionError):
            Simplex((1, 2)) <= (1, 2)


class TestSimplexClone:
    def test_clone_is_equal(self):
        s = Simplex((1, 2, 3))
        c = s.clone()
        assert s == c

    def test_clone_is_independent(self):
        s = Simplex((1, 2, 3))
        c = s.clone()
        assert s is not c


class TestSimplexRepr:
    def test_repr(self):
        s = Simplex((1,))
        assert "Simplex" in repr(s)

    def test_str(self):
        s = Simplex((1,))
        assert "Nodes" in str(s)
