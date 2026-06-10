"""Tests for 2D Pachner moves."""

import pytest

from mantra.augmentations.triangulation_2d import Triangulation2D


# Boundary of tetrahedron = minimal S^2 (4 vertices, 4 triangles)
S2_MINIMAL = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]

# Minimal torus triangulation (7 vertices, 14 triangles)
TORUS_MINIMAL = [
    [1, 2, 3],
    [1, 2, 4],
    [1, 3, 5],
    [1, 4, 6],
    [1, 5, 7],
    [1, 6, 7],
    [2, 3, 6],
    [2, 4, 7],
    [2, 5, 6],
    [2, 5, 7],
    [3, 4, 5],
    [3, 4, 7],
    [3, 6, 7],
    [4, 5, 6],
]


class TestTriangulation2DConstruction:
    def test_s2_minimal(self):
        t = Triangulation2D(S2_MINIMAL)
        assert t.n_vertices == 4
        assert len(t._simplices) == 4
        assert t.dimension == 2

    def test_to_list_roundtrip(self):
        t = Triangulation2D(S2_MINIMAL)
        result = t.to_list()
        assert len(result) == 4
        assert all(len(tri) == 3 for tri in result)

    def test_euler_characteristic_s2(self):
        t = Triangulation2D(S2_MINIMAL)
        assert t.euler_characteristic() == 2

    def test_euler_characteristic_torus(self):
        t = Triangulation2D(TORUS_MINIMAL)
        assert t.euler_characteristic() == 0

    def test_validate_s2(self):
        t = Triangulation2D(S2_MINIMAL)
        t.validate()  # should not raise

    def test_validate_torus(self):
        t = Triangulation2D(TORUS_MINIMAL)
        t.validate()

    def test_f_vector_s2(self):
        t = Triangulation2D(S2_MINIMAL)
        assert t.f_vector() == (4, 6, 4)

    def test_f_vector_torus(self):
        t = Triangulation2D(TORUS_MINIMAL)
        assert t.f_vector() == (7, 21, 14)


class TestFlipEdge:
    def test_flip_preserves_closed_manifold(self):
        t = Triangulation2D(S2_MINIMAL)
        result = t.flip_edge()
        if result:
            t.validate()

    def test_flip_preserves_euler_characteristic(self):
        t = Triangulation2D(S2_MINIMAL)
        chi_before = t.euler_characteristic()
        t.flip_edge()
        assert t.euler_characteristic() == chi_before

    def test_flip_preserves_triangle_count(self):
        """A 2-2 move should keep the number of triangles the same."""
        t = Triangulation2D(S2_MINIMAL)
        n_before = len(t._simplices)
        t.flip_edge()
        assert len(t._simplices) == n_before

    def test_flip_preserves_vertex_count(self):
        t = Triangulation2D(S2_MINIMAL)
        n_before = t.n_vertices
        t.flip_edge()
        assert t.n_vertices == n_before

    def test_flip_on_torus(self):
        t = Triangulation2D(TORUS_MINIMAL)
        chi_before = t.euler_characteristic()
        result = t.flip_edge()
        if result:
            t.validate()
            assert t.euler_characteristic() == chi_before


class TestSubdivide:
    def test_subdivide_increases_vertices(self):
        t = Triangulation2D(S2_MINIMAL)
        t.subdivide()
        assert t.n_vertices == 5

    def test_subdivide_increases_triangles(self):
        """1-3 move: 1 triangle becomes 3, net +2."""
        t = Triangulation2D(S2_MINIMAL)
        t.subdivide()
        assert len(t._simplices) == 6

    def test_subdivide_preserves_euler_characteristic(self):
        t = Triangulation2D(S2_MINIMAL)
        chi_before = t.euler_characteristic()
        t.subdivide()
        assert t.euler_characteristic() == chi_before

    def test_subdivide_preserves_closed_manifold(self):
        t = Triangulation2D(S2_MINIMAL)
        t.subdivide()
        t.validate()


class TestGlueTorus:
    def test_glue_torus_changes_euler_characteristic(self):
        """Gluing a torus decreases chi by 2."""
        t = Triangulation2D(S2_MINIMAL)
        chi_before = t.euler_characteristic()
        t.glue_torus()
        assert t.euler_characteristic() == chi_before - 2

    def test_glue_torus_preserves_closed_manifold(self):
        t = Triangulation2D(S2_MINIMAL)
        t.glue_torus()
        t.validate()

    def test_glue_torus_increases_vertices(self):
        t = Triangulation2D(S2_MINIMAL)
        n_before = t.n_vertices
        t.glue_torus()
        assert t.n_vertices == n_before + 4


class TestGlueCrosscap:
    def test_glue_crosscap_changes_euler_characteristic(self):
        """Gluing a crosscap decreases chi by 1."""
        t = Triangulation2D(S2_MINIMAL)
        chi_before = t.euler_characteristic()
        t.glue_crosscap()
        assert t.euler_characteristic() == chi_before - 1

    def test_glue_crosscap_preserves_closed_manifold(self):
        t = Triangulation2D(S2_MINIMAL)
        t.glue_crosscap()
        t.validate()

    def test_glue_crosscap_increases_vertices(self):
        t = Triangulation2D(S2_MINIMAL)
        n_before = t.n_vertices
        t.glue_crosscap()
        assert t.n_vertices == n_before + 3


class TestRandomPachnerSequence:
    @pytest.mark.parametrize(
        "triangulation,chi",
        [(S2_MINIMAL, 2), (TORUS_MINIMAL, 0)],
        ids=["S2", "T2"],
    )
    def test_sequence_preserves_invariants(self, triangulation, chi):
        """Apply 20 random Pachner moves, check invariants after
        each."""
        t = Triangulation2D(triangulation)
        for _ in range(20):
            t.random_pachner_move()
            assert t.euler_characteristic() == chi
            t.validate()
