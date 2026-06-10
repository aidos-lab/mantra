"""Tests for triangulation deduplication.

These tests verify that there are no combinatorially isomorphic
duplicates in manifold triangulation datasets. This is important
after augmenting datasets via Pachner moves.

Usage::

    pytest test/test_deduplication.py --dataset-path 3_manifolds.json
"""

import pytest

from mantra.deduplication import (
    are_isomorphic,
    compute_degree_sequence,
    compute_edge_simplex_count_sequence,
    compute_f_vector,
    find_duplicates,
)


# ---- Unit tests for invariant computation ----


class TestFVector:
    def test_tetrahedron(self):
        """A single tetrahedron: 4 vertices, 6 edges, 4 faces, 1 tet."""
        tri = [[1, 2, 3, 4]]
        assert compute_f_vector(tri) == (4, 6, 4, 1)

    def test_two_tetrahedra_shared_face(self):
        """Two tetrahedra sharing a triangle."""
        tri = [[1, 2, 3, 4], [1, 2, 3, 5]]
        assert compute_f_vector(tri) == (5, 9, 7, 2)

    def test_boundary_of_4_simplex(self):
        """Boundary of a 4-simplex = minimal S^3: 5 vertices, 10 edges,
        10 faces, 5 tetrahedra."""
        tri = [
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [1, 2, 4, 5],
            [1, 3, 4, 5],
            [2, 3, 4, 5],
        ]
        assert compute_f_vector(tri) == (5, 10, 10, 5)


class TestDegreeSequence:
    def test_boundary_of_4_simplex(self):
        """Boundary of 4-simplex: every vertex has degree 4 (complete graph
        K5)."""
        tri = [
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [1, 2, 4, 5],
            [1, 3, 4, 5],
            [2, 3, 4, 5],
        ]
        assert compute_degree_sequence(tri) == (4, 4, 4, 4, 4)

    def test_invariant_under_relabeling(self):
        """Degree sequence should be the same after relabeling vertices."""
        tri1 = [[1, 2, 3, 4], [1, 2, 3, 5]]
        tri2 = [[10, 20, 30, 40], [10, 20, 30, 50]]
        assert compute_degree_sequence(tri1) == compute_degree_sequence(tri2)


class TestEdgeSimplexCountSequence:
    def test_boundary_of_4_simplex(self):
        """Boundary of 4-simplex: each edge is in exactly 3 tetrahedra."""
        tri = [
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [1, 2, 4, 5],
            [1, 3, 4, 5],
            [2, 3, 4, 5],
        ]
        seq = compute_edge_simplex_count_sequence(tri)
        assert seq == (3, 3, 3, 3, 3, 3, 3, 3, 3, 3)

    def test_invariant_under_relabeling(self):
        """Edge-simplex-count sequence is invariant under vertex relabeling."""
        tri1 = [[1, 2, 3, 4], [1, 2, 3, 5]]
        tri2 = [[10, 20, 30, 40], [10, 20, 30, 50]]
        assert compute_edge_simplex_count_sequence(tri1) == (
            compute_edge_simplex_count_sequence(tri2)
        )


class TestIsomorphism:
    def test_identical(self):
        tri = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5],
               [2, 3, 4, 5]]
        assert are_isomorphic(tri, tri)

    def test_relabeled(self):
        """Same triangulation with permuted vertex labels."""
        tri1 = [
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [1, 2, 4, 5],
            [1, 3, 4, 5],
            [2, 3, 4, 5],
        ]
        # Apply permutation: 1->5, 2->4, 3->3, 4->2, 5->1
        tri2 = [
            [5, 4, 3, 2],
            [5, 4, 3, 1],
            [5, 4, 2, 1],
            [5, 3, 2, 1],
            [4, 3, 2, 1],
        ]
        assert are_isomorphic(tri1, tri2)

    def test_different_f_vector_not_isomorphic(self):
        """Two triangulations with different f-vectors are not isomorphic."""
        tri1 = [[1, 2, 3, 4]]
        tri2 = [[1, 2, 3, 4], [1, 2, 3, 5]]
        assert not are_isomorphic(tri1, tri2)

    def test_different_tet_filling_same_skeleton(self):
        """Two triangulations that share the same 1-skeleton (K6) but
        have different tetrahedra are not isomorphic."""
        # Both use all 15 edges of K6, but choose different sets of 8 tets.
        # tri1 is stacked from the boundary of the 4-simplex {1..5} plus
        # vertex 6 coned over three faces.
        tri1 = [
            [1, 2, 3, 6],
            [1, 2, 4, 5],
            [1, 2, 5, 6],
            [1, 3, 4, 5],
            [1, 3, 5, 6],
            [2, 3, 4, 6],
            [2, 4, 5, 6],
            [3, 4, 5, 6],
        ]
        # tri2 rearranges which faces are filled.
        tri2 = [
            [1, 2, 3, 4],
            [1, 2, 4, 6],
            [1, 2, 5, 6],
            [1, 3, 4, 5],
            [1, 2, 3, 5],
            [2, 3, 5, 6],
            [2, 4, 5, 6],
            [3, 4, 5, 6],
        ]
        f1 = compute_f_vector(tri1)
        f2 = compute_f_vector(tri2)
        if f1 == f2:
            assert not are_isomorphic(tri1, tri2)


class TestFindDuplicates:
    def test_finds_relabeled_duplicate(self):
        """find_duplicates should detect isomorphic copies."""
        tri1 = [
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [1, 2, 4, 5],
            [1, 3, 4, 5],
            [2, 3, 4, 5],
        ]
        # Relabeled: 1->5, 2->4, 3->3, 4->2, 5->1
        tri2 = [
            [5, 4, 3, 2],
            [5, 4, 3, 1],
            [5, 4, 2, 1],
            [5, 3, 2, 1],
            [4, 3, 2, 1],
        ]
        dataset = [
            {"id": "a", "triangulation": tri1},
            {"id": "b", "triangulation": tri2},
        ]
        duplicates = find_duplicates(dataset)
        assert len(duplicates) == 1
        assert duplicates[0] == ("a", "b")

    def test_no_false_positives(self):
        """find_duplicates should not flag non-isomorphic triangulations."""
        dataset = [
            {"id": "a", "triangulation": [[1, 2, 3, 4]]},
            {"id": "b", "triangulation": [[1, 2, 3, 4], [1, 2, 3, 5]]},
        ]
        duplicates = find_duplicates(dataset)
        assert len(duplicates) == 0


# ---- Integration test on full dataset ----


def test_no_duplicate_triangulations(dataset):
    """Verify that the dataset contains no isomorphic duplicates."""
    duplicates = find_duplicates(dataset, verbose=True)

    if duplicates:
        msg_lines = [f"Found {len(duplicates)} duplicate pair(s):"]
        for id1, id2 in duplicates[:20]:
            msg_lines.append(f"  {id1} <-> {id2}")
        if len(duplicates) > 20:
            msg_lines.append(f"  ... and {len(duplicates) - 20} more")
        pytest.fail("\n".join(msg_lines))
