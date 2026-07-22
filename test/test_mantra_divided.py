"""Tests for ``mantra.datasets.mantra_divided``."""

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
        with pytest.raises(ValueError, match="graded_vertex_number"):
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
                graded_vertex_number=10,
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
        ds_20 = make_divided(
            make_manifolds_json,
            balanced_entries,
            tmp_path,
            split_type="train",
            split_proportions=[0.2, 0.4, 0.4],
        )
        assert len(ds_60) == 6
        assert len(ds_20) == 2

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
            graded_vertex_number=9,
        )
        assert len(ds) > 0
        for d in ds:
            assert int(d.n_vertices) == 9


class TestMaxOODSizePerClass:
    def test_trimming_reduces_to_cap(self, make_manifolds_json, tmp_path):
        entries = [manifold_entry(f"s{i}") for i in range(10)]
        ds = make_divided(
            make_manifolds_json,
            entries,
            tmp_path,
            split_type="ood",
            division_type="graded",
            graded_vertex_number=12,
            split_proportions=[0.2, 0.2, 0.6],
            max_ood_size_per_class=2,
        )
        assert len(ds) == 2

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
            with pytest.raises(AssertionError):
                make_divided(
                    make_manifolds_json,
                    entries,
                    tmp_path,
                    split_type=split,
                    max_vertices=4,
                )

    def test_ood_strictly_larger_than_in_distribution(
        self, make_manifolds_json, tmp_path
    ):
        entries = [manifold_entry(f"t{i}") for i in range(5)] + [
            octahedron_entry(f"o{i}") for i in range(5)
        ]
        kwargs = dict(
            division_type="graded", graded_vertex_number=8, max_vertices=6
        )
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
        obj.max_vertices = kwargs.pop("max_vertices", None)
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
        names = self._names(
            division_type="graded",
            graded_vertex_number=50,
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
