"""Tests for the raw-triangulation subdivision utilities."""

import random
from itertools import combinations as combs

import pytest

from mantra.subdivision import (
    barycentric_stellar_graded,
    barycentric_subdivision_raw,
    stellar_subdivision_raw,
)


class TestBarycentricSubdivisionRaw:
    """Tests for barycentric_subdivision_raw on individual triangulations."""

    def test_single_triangle(self):
        """A single triangle [1,2,3] -> 6 triangles and 7 vertices."""
        tri = [[1, 2, 3]]
        new_tri, n_v = barycentric_subdivision_raw(tri)
        # 3 vertices + 3 edges + 1 face = 7 new vertices
        assert n_v == 7
        # 3! = 6 new triangles
        assert len(new_tri) == 6
        # 1,2,3 corners; 4,5,6 edge midpoints; 7 face barycenter
        assert new_tri == [
            [1, 4, 7],
            [1, 5, 7],
            [2, 4, 7],
            [2, 6, 7],
            [3, 5, 7],
            [3, 6, 7],
        ]

    def test_single_tetrahedron(self):
        """A single tetrahedron [1,2,3,4] -> 24 tetrahedra, 15 vertices."""
        tri = [[1, 2, 3, 4]]
        new_tri, n_v = barycentric_subdivision_raw(tri)
        # 4 vertices + 6 edges + 4 faces + 1 tetrahedron = 15 new vertices
        assert n_v == 15
        # 4! = 24 new tetrahedra
        assert len(new_tri) == 24

    def test_two_triangles_shared_edge(self):
        """Two triangles sharing an edge should share sub-simplex vertices."""
        tri = [[1, 2, 3], [2, 3, 4]]
        new_tri, n_v = barycentric_subdivision_raw(tri)

        # Vertices: {1}, {2}, {3}, {4} = 4
        # Edges: {1,2}, {1,3}, {2,3}, {2,4}, {3,4} = 5
        # Faces: {1,2,3}, {2,3,4} = 2
        # Total = 11
        assert n_v == 11
        # 2 triangles * 6 = 12 new triangles
        assert len(new_tri) == 12

        # 1,2,3,8 are the original vertices, 6 is barycenter of the original shared edge and 1,11 are the barycenters of the original triangles
        assert new_tri == [
            [1, 4, 7],
            [1, 5, 7],
            [2, 4, 7],
            [2, 6, 7],
            [2, 6, 11],
            [2, 9, 11],
            [3, 5, 7],
            [3, 6, 7],
            [3, 6, 11],
            [3, 10, 11],
            [8, 9, 11],
            [8, 10, 11],
        ]

    def test_vertices_are_one_indexed(self):
        """All vertex indices should be >= 1."""
        tri = [[1, 2, 3]]
        new_tri, _ = barycentric_subdivision_raw(tri)
        for simplex in new_tri:
            for v in simplex:
                assert v >= 1

    def test_simplex_dimension_preserved(self):
        """Output simplices keep the input number of vertices."""
        # 2D: triangles (3 vertices each)
        tri_2d = [[1, 2, 3], [2, 3, 4]]
        new_tri, _ = barycentric_subdivision_raw(tri_2d)
        for s in new_tri:
            assert len(s) == 3

        # 3D: tetrahedra (4 vertices each)
        tri_3d = [[1, 2, 3, 4]]
        new_tri, _ = barycentric_subdivision_raw(tri_3d)
        for s in new_tri:
            assert len(s) == 4

    def test_no_duplicate_simplices(self):
        """There should be no duplicate simplices in the output."""
        tri = [[1, 2, 3], [2, 3, 4]]
        new_tri, _ = barycentric_subdivision_raw(tri)
        as_tuples = [tuple(s) for s in new_tri]
        assert len(as_tuples) == len(set(as_tuples))

    def test_no_duplicate_simplices_3d(self):
        """There should be no duplicate simplices in the output."""
        tri = [[1, 2, 3, 4], [2, 3, 4, 5]]
        new_tri, _ = barycentric_subdivision_raw(tri)
        as_tuples = [tuple(s) for s in new_tri]
        assert len(as_tuples) == len(set(as_tuples))

    def test_empty(self):
        """An empty triangulation returns ([], 0) to accord with 0 vertices."""
        assert barycentric_subdivision_raw([]) == ([], 0)

    def test_double_subdivision(self):
        """Applying subdivision twice should produce correct counts."""
        tri = [[1, 2, 3]]
        # First subdivision: 6 triangles, 7 vertices
        new_tri, n_v = barycentric_subdivision_raw(tri)
        assert len(new_tri) == 6
        assert n_v == 7

        # Second subdivision: each of 6 triangles -> 6 new ones = 36
        new_tri2, n_v2 = barycentric_subdivision_raw(new_tri)
        assert len(new_tri2) == 36
        assert n_v2 > n_v

    def test_boundary_triangulation(self):
        """Boundary of a tetrahedron (4 triangles forming a sphere)."""
        tri = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        new_tri, n_v = barycentric_subdivision_raw(tri)
        # 4 vertices + 6 edges + 4 faces = 14 vertices
        assert n_v == 14
        # 4 * 6 = 24 new triangles
        assert len(new_tri) == 24

    def test_euler_characteristic_preserved_2d(self):
        """Subdivision preserves the Euler characteristic for 2-manifolds."""

        def euler_char(triangulation):
            vertices = set()
            edges = set()
            faces = set()
            for simplex in triangulation:
                s = tuple(sorted(simplex))
                faces.add(s)
                for e in combs(s, 2):
                    edges.add(e)
                for v in s:
                    vertices.add((v,))
            return len(vertices) - len(edges) + len(faces)

        # Boundary of tetrahedron = S^2, chi = 2
        tri = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        chi_original = euler_char(tri)
        assert chi_original == 2

        new_tri, _ = barycentric_subdivision_raw(tri)
        chi_subdivided = euler_char(new_tri)
        assert chi_subdivided == chi_original

    def test_euler_characteristic_preserved_3d(self):
        """Subdivision preserves Euler characteristic for 3-manifolds.

        For closed 3-manifolds, chi = 0.
        """

        def euler_char_3d(triangulation):
            cells = {0: set(), 1: set(), 2: set(), 3: set()}
            for simplex in triangulation:
                s = tuple(sorted(simplex))
                cells[3].add(s)
                for f in combs(s, 3):
                    cells[2].add(f)
                for e in combs(s, 2):
                    cells[1].add(e)
                for v in s:
                    cells[0].add((v,))
            return sum((-1) ** k * len(cells[k]) for k in range(4))

        # Boundary of a 5-cell (4-simplex) with vertices 1..5
        # gives 5 tetrahedra forming S^3
        tri_s3 = [
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [1, 2, 4, 5],
            [1, 3, 4, 5],
            [2, 3, 4, 5],
        ]
        chi_original = euler_char_3d(tri_s3)
        assert chi_original == 0, f"Expected chi=0, got {chi_original}"

        new_tri, _ = barycentric_subdivision_raw(tri_s3)
        chi_subdivided = euler_char_3d(new_tri)
        assert chi_subdivided == chi_original


class TestStellarSubdivisionRaw:
    """Tests for stellar_subdivision_raw (1-(d+1) Pachner moves)."""

    def test_full_single_triangle(self):
        """fraction=1.0 splits a triangle into 3 (one new barycenter)."""
        new_tri, n_v = stellar_subdivision_raw([[1, 2, 3]], fraction=1.0)
        assert n_v == 4
        assert len(new_tri) == 3
        for s in new_tri:
            assert len(s) == 3
            assert 4 in s  # every child contains the new barycenter

    def test_partial_with_rng(self):
        """0 < fraction < 1 subdivides a subset, passes the rest through."""
        tri = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        rng = random.Random(0)
        new_tri, n_v = stellar_subdivision_raw(tri, fraction=0.5, rng=rng)
        # 2 simplices subdivided (3 children each) + 2 passed through.
        assert len(new_tri) == 8
        # Two new barycenters added to the 4 original vertices.
        assert n_v == 6

    def test_partial_without_rng(self):
        """fraction < 1 with rng=None uses a fresh Random; counts hold."""
        tri = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        new_tri, n_v = stellar_subdivision_raw(tri, fraction=0.5)
        assert len(new_tri) == 8
        assert n_v == 6

    def test_empty(self):
        """An empty triangulation returns ([], 0)."""
        assert stellar_subdivision_raw([]) == ([], 0)

    def test_fraction_out_of_range(self):
        """fraction outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError):
            stellar_subdivision_raw([[1, 2, 3]], fraction=1.5)


class TestBarycentricStellarGraded:
    """Tests for barycentric_stellar_graded (subdivide to a vertex target)."""

    def test_graded(self):
        tri = [[1, 2, 3]]
        # Graded subdivision selects simplices at random, so pin the seed to
        # assert the exact output
        new_tri, n_v = barycentric_stellar_graded(tri, 5, rng=random.Random(0))
        assert n_v == 5
        assert len(new_tri) == 5

        # 1,2,3 are the original vertices, 4 is the barycenter of the original
        # triangle, and 5 is the barycenter of the new triangle on 1,3,4.
        assert new_tri == [
            [1, 2, 4],
            [1, 3, 5],
            [1, 4, 5],
            [2, 3, 4],
            [3, 4, 5],
        ]

    def test_graded_2(self):
        tri = [[1, 2, 3], [1, 2, 4]]
        new_tri, n_v = barycentric_stellar_graded(tri, 7, rng=random.Random(0))
        assert n_v == 7
        assert len(new_tri) == 8
        assert new_tri == [
            [1, 2, 6],
            [1, 2, 7],
            [1, 3, 6],
            [1, 4, 5],
            [1, 5, 7],
            [2, 3, 6],
            [2, 4, 5],
            [2, 5, 7],
        ]

    def test_graded_3d(self):
        tri = [[1, 2, 3, 4]]
        new_tri, n_v = barycentric_stellar_graded(tri, 6, rng=random.Random(0))
        assert n_v == 6
        assert len(new_tri) == 7

        # 1,2,3 are the original corners, 5 is is the first barycenter, 6 is the second one within the [1,2,3,5] tetrahedron
        assert new_tri == [
            [1, 2, 3, 5],
            [1, 2, 4, 5],
            [1, 3, 4, 5],
            [2, 3, 4, 6],
            [2, 3, 5, 6],
            [2, 4, 5, 6],
            [3, 4, 5, 6],
        ]

    def test_graded_3d_2(self):
        tri = [[1, 2, 3, 4], [1, 2, 3, 5]]
        new_tri, n_v = barycentric_stellar_graded(tri, 8, rng=random.Random(0))
        assert n_v == 8
        assert len(new_tri) == 11

    def test_graded_empty(self):
        """An empty triangulation returns ([], 0)."""
        assert barycentric_stellar_graded([], 5) == ([], 0)

    def test_graded_target_at_or_below_current_raises(self):
        """A target not above the current vertex count is rejected."""
        with pytest.raises(ValueError):
            barycentric_stellar_graded([[1, 2, 3]], 3, rng=random.Random(0))
        with pytest.raises(ValueError):
            barycentric_stellar_graded([[1, 2, 3]], 2, rng=random.Random(0))


class TestPureInputValidation:
    """Subdivision rejects non-pure (mixed-dimension) triangulations."""

    def test_mixed_simplex_sizes_raise(self):
        with pytest.raises(ValueError):
            barycentric_subdivision_raw([[1, 2, 3], [3, 4]])

    def test_stellar_mixed_simplex_sizes_raise(self):
        with pytest.raises(ValueError):
            stellar_subdivision_raw([[1, 2, 3], [3, 4]])
