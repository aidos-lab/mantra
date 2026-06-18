"""Tests for ``mantra.augmentations.base.Triangulation``."""

import random

import pytest

from mantra.augmentations.base import Triangulation


def make(top_simplices, dimension=2, rng=None):
    return Triangulation(top_simplices, dimension=dimension, rng=rng)


class TestConstruction:
    def test_stores_simplices_as_frozensets(self):
        t = make([[1, 2, 3]])
        assert t._simplices == {frozenset({1, 2, 3})}

    def test_next_vertex_is_max_plus_one(self):
        t = make([[1, 2, 3]])
        assert t._new_vertex() == 4

    def test_default_rng_is_module_random(self):
        t = make([[1, 2, 3]])
        assert t._rng is random

    def test_explicit_rng_is_used(self):
        rng = random.Random(0)
        t = make([[1, 2, 3]], rng=rng)
        assert t._rng is rng


class TestProperties:
    def test_dimension(self):
        assert make([[1, 2, 3]], dimension=2).dimension == 2

    def test_vertices(self):
        assert make([[1, 2, 3], [2, 3, 4]]).vertices == {1, 2, 3, 4}

    def test_n_vertices(self):
        assert make([[1, 2, 3], [2, 3, 4]]).n_vertices == 4


class TestNewVertex:
    def test_allocates_increasing_labels(self):
        t = make([[1, 2, 3]])
        assert (t._new_vertex(), t._new_vertex()) == (4, 5)
        assert t._next_vertex == 6


class TestToList:
    def test_sorted_and_canonical(self):
        t = make([[3, 1, 2]])
        assert t.to_list() == [[1, 2, 3]]

    def test_compacts_vertex_label_gaps(self):
        # Vertex 4 is missing -> labels remapped to a contiguous range,
        # so vertex 5 becomes 4 and max(label) == n_vertices.
        t = make([[1, 2, 3], [1, 2, 5]])
        assert t.to_list() == [[1, 2, 3], [1, 2, 4]]


class TestFaceToCofaces:
    def test_edges_of_single_triangle(self):
        t = make([[1, 2, 3]])
        cofaces = t.face_to_cofaces(1)
        assert cofaces[frozenset({1, 2})] == [frozenset({1, 2, 3})]
        assert set(cofaces) == {
            frozenset({1, 2}),
            frozenset({1, 3}),
            frozenset({2, 3}),
        }

    def test_shared_edge_has_two_cofaces(self):
        t = make([[1, 2, 3], [2, 3, 4]])
        cofaces = t.face_to_cofaces(1)
        assert len(cofaces[frozenset({2, 3})]) == 2


class TestFVectorAndEuler:
    def test_f_vector_triangle(self):
        assert make([[1, 2, 3]]).f_vector() == (3, 3, 1)

    def test_euler_characteristic_triangle(self):
        assert make([[1, 2, 3]]).euler_characteristic() == 1

    def test_euler_characteristic_sphere(self):
        # Boundary of a tetrahedron is a 2-sphere: chi = 2.
        sphere = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        assert make(sphere).euler_characteristic() == 2


class TestValidate:
    def test_closed_manifold_passes(self):
        # Tetrahedron boundary: every edge has exactly 2 triangles.
        sphere = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        make(sphere).validate()  # must not raise

    def test_boundary_face_raises(self):
        # A single triangle has edges with only one coface.
        with pytest.raises(ValueError, match="expected 2"):
            make([[1, 2, 3]]).validate()
