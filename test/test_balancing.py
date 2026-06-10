"""Tests for dataset balancing."""

from mantra.augmentations.balancing import (
    balance_dataset,
    _augment_triangulation,
)
from collections import Counter


# Minimal S^2 (boundary of tetrahedron)
S2_ENTRY = {
    "id": "manifold_2_4_1",
    "triangulation": [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]],
    "dimension": 2,
    "n_vertices": 4,
    "betti_numbers": [1, 0, 1],
    "name": "S^2",
    "orientable": True,
    "genus": 0,
}

# Minimal torus
T2_ENTRY = {
    "id": "manifold_2_7_1",
    "triangulation": [
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
    ],
    "dimension": 2,
    "n_vertices": 7,
    "betti_numbers": [1, 2, 1],
    "name": "T^2",
    "orientable": True,
    "genus": 1,
}

# Minimal S^3 (boundary of 4-simplex)
S3_ENTRY = {
    "id": "manifold_3_5_1",
    "triangulation": [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 4, 5],
        [1, 3, 4, 5],
        [2, 3, 4, 5],
    ],
    "dimension": 3,
    "n_vertices": 5,
    "betti_numbers": [1, 0, 0, 1],
    "name": "S^3",
}


class TestAugmentTriangulation:
    def test_2d_augment_preserves_name(self):
        result = _augment_triangulation(S2_ENTRY, 2, n_moves=3)
        assert result["name"] == "S^2"

    def test_2d_augment_changes_triangulation(self):
        result = _augment_triangulation(S2_ENTRY, 2, n_moves=5)
        # after 5 moves, triangulation should differ
        assert result["n_vertices"] >= 4

    def test_3d_augment_preserves_name(self):
        result = _augment_triangulation(S3_ENTRY, 3, n_moves=3)
        assert result["name"] == "S^3"

    def test_augment_updates_n_vertices(self):
        result = _augment_triangulation(S2_ENTRY, 2, n_moves=3)
        # n_vertices should match actual vertices in triangulation
        actual_verts = set()
        for tri in result["triangulation"]:
            actual_verts.update(tri)
        assert result["n_vertices"] == len(actual_verts)

    def test_augment_does_not_modify_original(self):
        original_tri = [list(t) for t in S2_ENTRY["triangulation"]]
        _augment_triangulation(S2_ENTRY, 2, n_moves=3)
        assert S2_ENTRY["triangulation"] == original_tri


class TestBalanceDataset:
    def test_balances_2d_classes(self):
        """Two classes, one with 10 S^2, one with 2 T^2."""
        dataset = [{**S2_ENTRY, "id": f"s2_{i}"} for i in range(10)] + [
            {**T2_ENTRY, "id": f"t2_{i}"} for i in range(2)
        ]

        result = balance_dataset(
            dataset,
            dimension=2,
            target_count=5,
            use_topology_changes=False,
        )

        counts = Counter(e["name"] for e in result)
        assert counts["S^2"] == 5
        assert counts["T^2"] == 5

    def test_balances_3d_classes(self):
        dataset = [{**S3_ENTRY, "id": f"s3_{i}"} for i in range(8)]
        result = balance_dataset(dataset, dimension=3, target_count=5)
        assert len(result) == 5

    def test_preserves_metadata(self):
        dataset = [{**S2_ENTRY, "id": "s2_0"}]
        result = balance_dataset(
            dataset,
            dimension=2,
            target_count=3,
            use_topology_changes=False,
        )
        for entry in result:
            assert entry["name"] == "S^2"
            assert entry["dimension"] == 2
            assert "triangulation" in entry

    def test_unique_ids(self):
        dataset = [{**S2_ENTRY, "id": "s2_0"}]
        result = balance_dataset(
            dataset,
            dimension=2,
            target_count=5,
            use_topology_changes=False,
        )
        ids = [e["id"] for e in result]
        assert len(ids) == len(set(ids))

    def test_topology_change_generates_new_class(self):
        """Gluing torus to S^2 should create T^2 samples."""
        dataset = [{**S2_ENTRY, "id": f"s2_{i}"} for i in range(5)]
        result = balance_dataset(
            dataset,
            dimension=2,
            target_count=3,
            use_topology_changes=True,
        )
        names = {e["name"] for e in result}
        # T^2 should be generated from S^2 via torus gluing
        assert "T^2" in names

    def test_reproducible_with_seed(self):
        dataset = [{**S2_ENTRY, "id": f"s2_{i}"} for i in range(3)]
        r1 = balance_dataset(
            dataset,
            dimension=2,
            target_count=5,
            seed=42,
            use_topology_changes=False,
        )
        r2 = balance_dataset(
            dataset,
            dimension=2,
            target_count=5,
            seed=42,
            use_topology_changes=False,
        )
        t1 = [e["triangulation"] for e in r1]
        t2 = [e["triangulation"] for e in r2]
        assert t1 == t2

    def test_max_vertices_filters_input(self):
        """Entries exceeding max_vertices are excluded before balancing."""
        dataset = [
            {**S2_ENTRY, "id": "s2_0"},  # 4 vertices
            {**T2_ENTRY, "id": "t2_0"},  # 7 vertices
        ]
        result = balance_dataset(
            dataset,
            dimension=2,
            target_count=1,
            max_vertices=5,
            use_topology_changes=False,
        )
        names = {e["name"] for e in result}
        assert "T^2" not in names
        assert "S^2" in names

    def test_max_vertices_filters_augmented(self):
        """Augmented entries exceeding max_vertices are discarded."""
        dataset = [{**S2_ENTRY, "id": "s2_0"}]
        result = balance_dataset(
            dataset,
            dimension=2,
            target_count=20,
            n_moves=10,
            max_vertices=4,
            use_topology_changes=False,
        )
        for entry in result:
            assert entry["n_vertices"] <= 4

    def test_max_vertices_respected_with_dedup(self):
        """Vertex limit is enforced even when dedup loop regenerates."""
        dataset = [{**S2_ENTRY, "id": "s2_0"}]
        result = balance_dataset(
            dataset,
            dimension=2,
            target_count=10,
            n_moves=3,
            max_vertices=6,
            use_topology_changes=False,
            dedup_max_rounds=5,
        )
        for entry in result:
            assert entry["n_vertices"] <= 6

    def test_max_vertices_respected_without_dedup(self):
        """Vertex limit is enforced even when dedup is disabled."""
        dataset = [{**S2_ENTRY, "id": "s2_0"}]
        result = balance_dataset(
            dataset,
            dimension=2,
            target_count=10,
            n_moves=3,
            max_vertices=6,
            use_topology_changes=False,
            dedup_max_rounds=0,
        )
        for entry in result:
            assert entry["n_vertices"] <= 6

    def test_max_vertices_none_is_noop(self):
        """max_vertices=None should not filter anything."""
        dataset = [
            {**S2_ENTRY, "id": "s2_0"},
            {**T2_ENTRY, "id": "t2_0"},
        ]
        result = balance_dataset(
            dataset,
            dimension=2,
            target_count=1,
            max_vertices=None,
            use_topology_changes=False,
        )
        names = {e["name"] for e in result}
        assert "S^2" in names
        assert "T^2" in names
