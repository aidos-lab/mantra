"""Tests for ``mantra.deduplication``."""

from mantra.deduplication import (
    are_isomorphic,
    compute_degree_sequence,
    compute_edge_simplex_count_sequence,
    compute_f_vector,
    compute_invariant_key,
    compute_wl_hash,
    find_duplicates,
)

TRIANGLE = [[1, 2, 3]]
# Same complex, relabelled vertices -> isomorphic to TRIANGLE.
TRIANGLE_RELABELLED = [[4, 5, 6]]
# Two triangles sharing an edge -> different f-vector from TRIANGLE.
TWO_TRIANGLES = [[1, 2, 3], [1, 2, 4]]

# Two non-isomorphic trees with the same degree sequence
# (1,1,1,1,2,3,3) -> identical invariant key, but the WL hash of the
# incidence graph tells them apart (so they land in singleton WL
# subgroups).
TREE_A = [[1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [6, 7]]
TREE_B = [[1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [5, 7]]

# K_{3,3} and the triangular prism: both 3-regular on 6 vertices, so
# they share an invariant key *and* a WL hash (regular incidence graphs
# are WL-indistinguishable) yet are not isomorphic. This forces the VF2
# safety net to reject a WL-hash collision.
K33 = [[1, 4], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6]]
PRISM = [
    [1, 2],
    [2, 3],
    [1, 3],
    [4, 5],
    [5, 6],
    [4, 6],
    [1, 4],
    [2, 5],
    [3, 6],
]


class TestInvariants:
    def test_f_vector_triangle(self):
        assert compute_f_vector(TRIANGLE) == (3, 3, 1)

    def test_f_vector_3d(self):
        assert compute_f_vector([[1, 2, 3, 4]]) == (4, 6, 4, 1)

    def test_degree_sequence_triangle(self):
        assert compute_degree_sequence(TRIANGLE) == (2, 2, 2)

    def test_edge_simplex_count_sequence(self):
        assert compute_edge_simplex_count_sequence(TRIANGLE) == (1, 1, 1)

    def test_edge_simplex_count_shared_edge(self):
        # Edge {1, 2} is shared by both triangles -> count 2.
        assert compute_edge_simplex_count_sequence(TWO_TRIANGLES) == (
            1,
            1,
            1,
            1,
            2,
        )

    def test_invariant_key_bundles_three_invariants(self):
        key = compute_invariant_key(TRIANGLE)
        assert key == (
            compute_f_vector(TRIANGLE),
            compute_degree_sequence(TRIANGLE),
            compute_edge_simplex_count_sequence(TRIANGLE),
        )

    def test_relabelling_preserves_invariant_key(self):
        assert compute_invariant_key(TRIANGLE) == compute_invariant_key(
            TRIANGLE_RELABELLED
        )


class TestWlHash:
    def test_isomorphic_have_equal_hash(self):
        assert compute_wl_hash(TRIANGLE) == compute_wl_hash(
            TRIANGLE_RELABELLED
        )

    def test_non_isomorphic_have_different_hash(self):
        assert compute_wl_hash(TRIANGLE) != compute_wl_hash(TWO_TRIANGLES)


class TestAreIsomorphic:
    def test_relabelled_is_isomorphic(self):
        assert are_isomorphic(TRIANGLE, TRIANGLE_RELABELLED) is True

    def test_different_complexes_not_isomorphic(self):
        assert are_isomorphic(TRIANGLE, TWO_TRIANGLES) is False


class TestFindDuplicates:
    def _entries(self, triangulations):
        return [
            {"id": i, "triangulation": t} for i, t in enumerate(triangulations)
        ]

    def test_detects_isomorphic_duplicate_via_vf2(self):
        data = self._entries([TRIANGLE, TRIANGLE_RELABELLED])
        assert find_duplicates(data) == [(0, 1)]

    def test_no_duplicates_when_all_distinct(self):
        data = self._entries([TRIANGLE, TWO_TRIANGLES])
        assert find_duplicates(data) == []

    def test_singleton_invariant_groups_are_skipped(self):
        # Distinct invariants -> no group has >1 member.
        data = self._entries([TRIANGLE, [[1, 2, 3, 4]]])
        assert find_duplicates(data) == []

    def test_singleton_wl_subgroups_are_skipped(self):
        # Same invariant key but distinct WL hashes -> each WL subgroup
        # has a single member, so the pairwise check is skipped.
        data = self._entries([TREE_A, TREE_B])
        assert find_duplicates(data) == []

    def test_vf2_rejects_wl_hash_collision(self):
        # K_{3,3} and the prism collide on both the invariant key and
        # the WL hash; only the VF2 isomorphism check separates them.
        data = self._entries([K33, PRISM])
        assert find_duplicates(data) == []

    def test_skips_vf2_for_large_wl_subgroups(self):
        # iso_max_group_size=0 forces WL-hash-only deduplication for
        # every nontrivial subgroup (size > 0).
        data = self._entries([TRIANGLE, TRIANGLE_RELABELLED])
        dups = find_duplicates(data, iso_max_group_size=0)
        assert dups == [(0, 1)]

    def test_verbose_with_large_subgroup(self, capsys):
        # Exercises the verbose progress prints and the skip-VF2 path.
        data = self._entries([TRIANGLE, TRIANGLE_RELABELLED, TRIANGLE])
        dups = find_duplicates(data, verbose=True, iso_max_group_size=0)
        # Three isomorphic copies -> two duplicate pairs against the
        # first member.
        assert len(dups) == 2
        assert "WL subgroup" in capsys.readouterr().err

    def test_verbose_level1_progress(self, capsys):
        # The level-1 progress line fires every 10000 entries; this
        # also drives the large-group verbose deduplication path.
        data = self._entries([[[1, 2]]] * 10000)
        dups = find_duplicates(data, verbose=True)
        assert len(dups) == 9999
        assert "10000/10000" in capsys.readouterr().err
