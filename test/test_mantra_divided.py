"""Tests for ``mantra.datasets.mantra_divided``."""

import warnings
from collections import Counter

import pytest

from mantra.datasets import MANTRADivided
from mantra.datasets.mantra_divided import SubdivisionType

from .conftest import manifold_entry

# A 2-sphere triangulated as an octahedron (6 vertices, 8 triangles);
# a second, larger sphere triangulation next to the tetrahedral one
# from ``conftest`` (4 vertices).
OCTAHEDRON = [
    [1, 2, 3],
    [1, 3, 4],
    [1, 4, 5],
    [1, 2, 5],
    [2, 3, 6],
    [3, 4, 6],
    [4, 5, 6],
    [2, 5, 6],
]


def octahedron_entry(id, **extra):
    return manifold_entry(id, triangulation=OCTAHEDRON, n_vertices=6, **extra)


def make_divided(make_manifolds_json, entries, tmp_path, **kwargs):
    path = make_manifolds_json(entries)
    kwargs.setdefault("dimension", 2)
    return MANTRADivided(str(tmp_path / "root"), local_path=path, **kwargs)


class TestSubdivisionType:
    def test_roundtrip(self):
        for sub in SubdivisionType:
            assert SubdivisionType.from_str(str(sub)) == sub

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="no Subdivision"):
            SubdivisionType.from_str("septagonal")


class TestValidation:
    def test_invalid_split_type_raises(self, tmp_path):
        with pytest.raises(ValueError, match="split_type"):
            MANTRADivided(str(tmp_path / "root"), split_type="holdout")

    def test_invalid_split_proportions_raises(self, tmp_path):
        with pytest.raises(ValueError, match="split_proportions"):
            MANTRADivided(
                str(tmp_path / "root"),
                split_type="train",
                split_proportions=[0.5, 0.5, 0.5],
            )

    def test_graded_without_vertex_number_raises(self, tmp_path):
        with pytest.raises(ValueError, match="vertex_number"):
            MANTRADivided(
                str(tmp_path / "root"),
                split_type="ood",
                division_type="graded",
            )

    def test_graded_vertex_number_below_max_vertices_raises(self, tmp_path):
        with pytest.raises(ValueError, match="strictly greater"):
            MANTRADivided(
                str(tmp_path / "root"),
                split_type="ood",
                division_type="graded",
                vertex_number=10,
                max_vertices=10,
            )


class TestSplits:
    def test_split_sizes_follow_proportions(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        sizes = {}
        for split in ["train", "val", "test", "ood"]:
            ds = make_divided(
                make_manifolds_json,
                balanced_entries,
                tmp_path,
                split_type=split,
            )
            sizes[split] = len(ds)
        assert sizes == {"train": 6, "val": 2, "test": 2, "ood": 2}

    def test_splits_are_disjoint(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        ids = {}
        for split in ["train", "val", "test"]:
            ds = make_divided(
                make_manifolds_json,
                balanced_entries,
                tmp_path,
                split_type=split,
            )
            ids[split] = {d.id for d in ds}
        assert not ids["train"] & ids["val"]
        assert not ids["train"] & ids["test"]
        assert not ids["val"] & ids["test"]

    def test_changed_split_proportions_reprocess(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        # Different proportions must not silently reuse the cached
        # splits of a previous configuration under the same root.
        ds_60 = make_divided(
            make_manifolds_json,
            balanced_entries,
            tmp_path,
            split_type="train",
            split_proportions=[0.6, 0.2, 0.2],
        )
        ds_80 = make_divided(
            make_manifolds_json,
            balanced_entries,
            tmp_path,
            split_type="train",
            split_proportions=[0.8, 0.1, 0.1],
        )
        assert len(ds_60) == 6
        assert len(ds_80) == 8

    def test_stratified_split_balances_classes(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        for split, expected in [("train", 3), ("val", 1), ("test", 1)]:
            ds = make_divided(
                make_manifolds_json,
                balanced_entries,
                tmp_path,
                split_type=split,
                stratified=True,
            )
            counts = Counter(d.name for d in ds)
            assert counts == {"S^2": expected, "RP^2": expected}

    def test_class_count_filter_drops_rare_classes(
        self, make_manifolds_json, tmp_path
    ):
        entries = [manifold_entry(f"s{i}", name="S^2") for i in range(9)] + [
            manifold_entry("r0", name="RP^2")
        ]
        ds = make_divided(
            make_manifolds_json,
            entries,
            tmp_path,
            split_type="train",
            class_count_filter=1,
        )
        assert all(d.name == "S^2" for d in ds)

    def test_pre_filter_and_pre_transform_applied(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        def pre_filter(d):
            return d.orientable

        def pre_transform(d):
            d.tagged = True
            return d

        ds = make_divided(
            make_manifolds_json,
            balanced_entries,
            tmp_path,
            split_type="ood",
            pre_filter=pre_filter,
            pre_transform=pre_transform,
        )
        assert all(d.tagged for d in ds)
        assert all(d.name == "S^2" for d in ds)


class TestOODSubdivision:
    def test_barycentric_default(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        ds = make_divided(
            make_manifolds_json,
            balanced_entries,
            tmp_path,
            split_type="ood",
        )
        # One barycentric round of the tetrahedral sphere:
        # 4 vertices + 6 edges + 4 faces = 14 vertices, 24 triangles.
        for d in ds:
            assert int(d.n_vertices) == 14
            assert len(d.triangulation) == 24
            assert d.id.endswith("_ood_0") or "_ood_" in d.id

    def test_barycentric_two_rounds(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        ds = make_divided(
            make_manifolds_json,
            balanced_entries,
            tmp_path,
            split_type="ood",
            round=2,
        )
        # Second round on (V=14, E=36, F=24): 14 + 36 + 24 = 74 vertices.
        for d in ds:
            assert int(d.n_vertices) == 74
            assert len(d.triangulation) == 144

    def test_stellar_full_fraction(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        ds = make_divided(
            make_manifolds_json,
            balanced_entries,
            tmp_path,
            split_type="ood",
            division_type="stellar",
            fraction=1.0,
        )
        # Every triangle of the tetrahedral sphere gains a barycenter:
        # 4 + 4 = 8 vertices, 4 * 3 = 12 triangles.
        for d in ds:
            assert int(d.n_vertices) == 8
            assert len(d.triangulation) == 12

    def test_graded_reaches_exact_vertex_number(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        ds = make_divided(
            make_manifolds_json,
            balanced_entries,
            tmp_path,
            split_type="ood",
            division_type="graded",
            vertex_number=9,
        )
        assert len(ds) > 0
        for d in ds:
            assert int(d.n_vertices) == 9

    def test_graded_drops_sources_at_or_above_vertex_number(
        self, make_manifolds_json, tmp_path
    ):
        # Octahedra (6 vertices) reach the target and must be excluded;
        # a proportionally large test split makes sure both kinds land
        # in it.
        entries = [manifold_entry(f"t{i}") for i in range(5)] + [
            octahedron_entry(f"o{i}") for i in range(5)
        ]
        with pytest.warns(UserWarning, match="Excluded"):
            ds = make_divided(
                make_manifolds_json,
                entries,
                tmp_path,
                split_type="ood",
                division_type="graded",
                vertex_number=6,
                split_proportions=[0.2, 0.2, 0.6],
            )
        assert len(ds) > 0
        for d in ds:
            assert int(d.n_vertices) == 6
            # only the 4-vertex tetrahedral spheres remain as sources
            assert d.id.startswith("t")


class TestMaxOODSizePerClass:
    def test_oversampling_reaches_cap(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        ds = make_divided(
            make_manifolds_json,
            balanced_entries,
            tmp_path,
            split_type="ood",
            stratified=True,
            division_type="graded",
            vertex_number=12,
            max_ood_size_per_class=5,
        )
        counts = Counter(d.name for d in ds)
        assert counts == {"S^2": 5, "RP^2": 5}
        # Oversampled graded subdivisions are randomized, so cycling the
        # same source must yield distinct triangulations.
        triangulations = {tuple(tuple(s) for s in d.triangulation) for d in ds}
        assert len(triangulations) > 2

    def test_trimming_reduces_to_cap(self, make_manifolds_json, tmp_path):
        entries = [manifold_entry(f"s{i}") for i in range(10)]
        ds = make_divided(
            make_manifolds_json,
            entries,
            tmp_path,
            split_type="ood",
            division_type="graded",
            vertex_number=12,
            split_proportions=[0.2, 0.2, 0.6],
            max_ood_size_per_class=2,
        )
        assert len(ds) == 2

    def test_barycentric_oversampling_warns_about_duplicates(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        with pytest.warns(UserWarning, match="duplicates"):
            ds = make_divided(
                make_manifolds_json,
                balanced_entries,
                tmp_path,
                split_type="ood",
                stratified=True,
                max_ood_size_per_class=3,
            )
        counts = Counter(d.name for d in ds)
        assert counts == {"S^2": 3, "RP^2": 3}

    def test_full_stellar_oversampling_warns_about_duplicates(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        # Stellar with fraction=1.0 subdivides every top simplex and is
        # just as deterministic as barycentric.
        with pytest.warns(UserWarning, match="duplicates"):
            make_divided(
                make_manifolds_json,
                balanced_entries,
                tmp_path,
                split_type="ood",
                stratified=True,
                division_type="stellar",
                fraction=1.0,
                max_ood_size_per_class=3,
            )

    def test_partial_stellar_oversampling_is_randomized(
        self, make_manifolds_json, balanced_entries, tmp_path
    ):
        ds = make_divided(
            make_manifolds_json,
            balanced_entries,
            tmp_path,
            split_type="ood",
            stratified=True,
            division_type="stellar",
            fraction=0.5,
            max_ood_size_per_class=4,
        )
        triangulations = {tuple(tuple(s) for s in d.triangulation) for d in ds}
        assert len(triangulations) > 1


class TestMaxVertices:
    def test_max_vertices_filters_in_distribution_splits(
        self, make_manifolds_json, tmp_path
    ):
        entries = [manifold_entry(f"t{i}") for i in range(5)] + [
            octahedron_entry(f"o{i}") for i in range(5)
        ]
        for split in ["train", "val", "test"]:
            ds = make_divided(
                make_manifolds_json,
                entries,
                tmp_path,
                split_type=split,
                max_vertices=4,
            )
            assert all(int(d.n_vertices) <= 4 for d in ds)

    def test_ood_strictly_larger_than_in_distribution(
        self, make_manifolds_json, tmp_path
    ):
        entries = [manifold_entry(f"t{i}") for i in range(5)] + [
            octahedron_entry(f"o{i}") for i in range(5)
        ]
        kwargs = dict(division_type="graded", vertex_number=8, max_vertices=6)
        max_in_dist = max(
            int(d.n_vertices)
            for split in ["train", "val", "test"]
            for d in make_divided(
                make_manifolds_json,
                entries,
                tmp_path,
                split_type=split,
                **kwargs,
            )
        )
        ood = make_divided(
            make_manifolds_json,
            entries,
            tmp_path,
            split_type="ood",
            **kwargs,
        )
        assert max_in_dist <= 6
        assert all(int(d.n_vertices) > max_in_dist for d in ood)


class TestProcessedFileNames:
    def _names(self, **kwargs):
        obj = MANTRADivided.__new__(MANTRADivided)
        obj.division_type = SubdivisionType.from_str(
            kwargs.pop("division_type", "barycentric")
        )
        obj.max_ood_size_per_class = kwargs.pop("max_ood_size_per_class", None)
        obj.class_count_filter = kwargs.pop("class_count_filter", None)
        obj.split_proportions = kwargs.pop(
            "split_proportions", [0.6, 0.2, 0.2]
        )
        obj.stratified = kwargs.pop("stratified", False)
        obj.kwargs = kwargs
        return obj.processed_file_names

    def test_default_names(self):
        assert self._names() == [
            "train.pt",
            "val.pt",
            "test.pt",
            "ood_barycentric_1.pt",
        ]

    def test_names_encode_parameters(self):
        # max_vertices needs no file-name entry: the parent class
        # encodes it into the processed directory.
        names = self._names(
            division_type="graded",
            vertex_number=50,
            max_ood_size_per_class=100,
            class_count_filter=5,
        )
        assert names == [
            "train_ccf5.pt",
            "val_ccf5.pt",
            "test_ccf5.pt",
            "ood_graded_50_cap100_ccf5.pt",
        ]

    def test_names_encode_split_proportions_and_stratified(self):
        names = self._names(split_proportions=[0.8, 0.1, 0.1], stratified=True)
        assert names[0] == "train_sp0.8-0.1-0.1_strat.pt"

    def test_stellar_name_encodes_fraction(self):
        names = self._names(division_type="stellar", fraction=0.5)
        assert names[-1] == "ood_stellar_0.5.pt"


class TestBalancedDivided:
    BALANCE_KWARGS = dict(
        target_count=4, n_moves=1, use_topology_changes=False
    )

    def test_balancing_feeds_the_splits(
        self, make_manifolds_json, tmp_path, no_dedup
    ):
        # Imbalanced input; balancing yields 4 per class = 8 entries,
        # which then split 5/1/2 with proportions [0.6, 0.2, 0.2].
        entries = [manifold_entry(f"s{i}", name="S^2") for i in range(6)] + [
            manifold_entry(f"r{i}", name="RP^2", orientable=False)
            for i in range(2)
        ]
        sizes = {}
        for split in ["train", "val", "test", "ood"]:
            ds = make_divided(
                make_manifolds_json,
                entries,
                tmp_path,
                split_type=split,
                balanced=True,
                balance_kwargs=self.BALANCE_KWARGS,
            )
            sizes[split] = len(ds)
        assert sizes["train"] + sizes["val"] + sizes["test"] == 8
        assert sizes["ood"] == sizes["test"]

    def test_processed_dir_separates_balanced(
        self, make_manifolds_json, balanced_entries, tmp_path, no_dedup
    ):
        plain = make_divided(
            make_manifolds_json,
            balanced_entries,
            tmp_path,
            split_type="train",
        )
        balanced = make_divided(
            make_manifolds_json,
            balanced_entries,
            tmp_path,
            split_type="train",
            balanced=True,
            balance_kwargs=self.BALANCE_KWARGS,
        )
        assert plain.processed_dir != balanced.processed_dir
        assert plain.processed_dir.endswith("unbalanced_42")
        assert balanced.processed_dir.endswith(
            "balanced_42_n_moves1_target_count4_use_topology_changesFalse"
        )

    def test_balanced_with_class_count_filter_warns(
        self, make_manifolds_json, balanced_entries, tmp_path, no_dedup
    ):
        with pytest.warns(UserWarning, match="re-imbalance"):
            make_divided(
                make_manifolds_json,
                balanced_entries,
                tmp_path,
                split_type="train",
                balanced=True,
                balance_kwargs=self.BALANCE_KWARGS,
                class_count_filter=1,
            )

    def test_balance_kwargs_max_vertices_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="top-level max_vertices"):
            MANTRADivided(
                str(tmp_path / "root"),
                split_type="train",
                balanced=True,
                balance_kwargs={"max_vertices": 5},
            )

    def test_max_vertices_forwarded_to_balancing(
        self, make_manifolds_json, tmp_path, no_dedup
    ):
        # Octahedral spheres (6 vertices) exceed the cap and must be
        # excluded inside the balancing; every class is then balanced
        # to target_count from the tetrahedral sources alone, and no
        # re-imbalance warning is emitted.
        entries = (
            [manifold_entry(f"s{i}", name="S^2") for i in range(3)]
            + [octahedron_entry(f"o{i}", name="S^2") for i in range(2)]
            + [
                manifold_entry(f"r{i}", name="RP^2", orientable=False)
                for i in range(2)
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = make_divided(
                make_manifolds_json,
                entries,
                tmp_path,
                split_type="train",
                balanced=True,
                balance_kwargs=self.BALANCE_KWARGS,
                max_vertices=5,
            )
        assert not [
            w for w in caught if "re-imbalance" in str(w.message)
        ]
        assert ds.max_vertices == 5
        assert all(int(d.n_vertices) <= 5 for d in ds)
