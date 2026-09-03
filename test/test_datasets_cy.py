"""Tests for the parquet-based CY dataset and its splits."""

from collections import Counter

import pytest
import torch

from mantra.datasets import CalabiYau, CalabiYauDataset


def _load(tmp_path, make_cy_parquet, cy_rows, **kwargs):
    return CalabiYau(
        root=str(tmp_path / "data"),
        local_path=make_cy_parquet(cy_rows),
        **kwargs,
    )


def _load_split(tmp_path, make_cy_parquet, rows, split_type, **kwargs):
    return CalabiYauDataset(
        root=str(tmp_path / "data"),
        split_type=split_type,
        local_path=make_cy_parquet(rows),
        **kwargs,
    )


class TestCY:
    def test_roundtrip(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows)

        assert len(dataset) == len(cy_rows)

        data = dataset[0]

        # Simplices are converted to the 1-indexed MANTRA convention,
        # `dimension` holds the topological dimension.
        assert data.triangulation == [
            [1, 2, 3],
            [1, 3, 4],
            [1, 4, 5],
            [1, 5, 2],
        ]
        assert int(data.dimension) == 2
        assert int(data.n_vertices) == 5

        assert data.coords.dtype == torch.float32
        assert data.coords.shape == (5, 2)

        # Extra parquet columns become attributes.
        assert int(data.h11) == 6
        assert int(data.h12) == 46
        assert int(dataset[1].h11) == 7

    def test_limit(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows, limit=1)

        assert len(dataset) == 1
        assert "limit_1" in dataset.processed_dir

        # The limited variant must not shadow the full dataset.
        full = _load(tmp_path, make_cy_parquet, cy_rows)
        assert len(full) == len(cy_rows)

    def test_name(self, tmp_path, make_cy_parquet, cy_rows):
        dataset = _load(tmp_path, make_cy_parquet, cy_rows, name="variant_a")

        assert "variant_a" in dataset.processed_dir
        assert len(dataset) == len(cy_rows)

        # The named variant lives in its own processed directory and
        # must not shadow the default dataset.
        full = _load(tmp_path, make_cy_parquet, cy_rows)
        assert "variant_a" not in full.processed_dir
        assert len(full) == len(cy_rows)


def _numbered_rows(n):
    """CY-style rows whose ``h11`` enumerates the samples and whose
    ``h12`` cycles through three values."""
    simplices = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
    vertices = [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
    ]
    return [
        {
            "simplices": simplices,
            "vertices": vertices,
            "h11": i,
            "h12": i % 3,
        }
        for i in range(n)
    ]


def _ids(dataset):
    return sorted(int(data.h11) for data in dataset)


class TestCYSplits:
    def test_splits_partition_the_dataset(self, tmp_path, make_cy_parquet):
        rows = _numbered_rows(20)
        splits = {
            split_type: _load_split(
                tmp_path, make_cy_parquet, rows, split_type
            )
            for split_type in ("train", "val", "test")
        }

        # Default proportions [0.6, 0.2, 0.2] of 20 samples.
        assert len(splits["train"]) == 12
        assert len(splits["val"]) == 4
        assert len(splits["test"]) == 4

        seen = [int(data.h11) for ds in splits.values() for data in ds]
        assert sorted(seen) == list(range(20))

    def test_splits_are_deterministic(self, tmp_path, make_cy_parquet):
        rows = _numbered_rows(20)
        first = _load_split(tmp_path, make_cy_parquet, rows, "val", seed=7)
        again = _load_split(
            tmp_path / "other_root", make_cy_parquet, rows, "val", seed=7
        )
        assert _ids(first) == _ids(again)

    def test_seed_changes_the_assignment(self, tmp_path, make_cy_parquet):
        rows = _numbered_rows(40)
        default = _load_split(tmp_path, make_cy_parquet, rows, "test", seed=42)
        reseeded = _load_split(
            tmp_path, make_cy_parquet, rows, "test", seed=43
        )
        # Differently seeded splits live in differently named files.
        assert default.processed_paths != reseeded.processed_paths
        assert _ids(default) != _ids(reseeded)

    def test_split_proportions_change_sizes(self, tmp_path, make_cy_parquet):
        rows = _numbered_rows(20)
        test = _load_split(
            tmp_path,
            make_cy_parquet,
            rows,
            "test",
            split_proportions=[0.5, 0.1, 0.4],
        )
        assert len(test) == 8

    def test_full_dataset_coexists_with_splits(
        self, tmp_path, make_cy_parquet
    ):
        rows = _numbered_rows(20)
        train = _load_split(tmp_path, make_cy_parquet, rows, "train")
        full = _load(tmp_path, make_cy_parquet, rows)
        assert len(train) == 12
        assert len(full) == 20

    def test_invalid_split_type_rejected(self, tmp_path, make_cy_parquet):
        with pytest.raises(ValueError, match="split_type"):
            _load_split(tmp_path, make_cy_parquet, _numbered_rows(5), "ood")

    def test_invalid_proportions_rejected(self, tmp_path, make_cy_parquet):
        with pytest.raises(ValueError, match="split_proportions"):
            _load_split(
                tmp_path,
                make_cy_parquet,
                _numbered_rows(5),
                "train",
                split_proportions=[0.8, 0.1, 0.2],
            )


class TestCYStratified:
    def test_stratified_split_balances_label_source(
        self, tmp_path, make_cy_parquet
    ):
        rows = _numbered_rows(30)
        counts = {
            split_type: Counter(
                int(data.h12)
                for data in _load_split(
                    tmp_path,
                    make_cy_parquet,
                    rows,
                    split_type,
                    stratified=True,
                    label_source="h12",
                )
            )
            for split_type in ("train", "val", "test")
        }
        assert counts["train"] == {0: 6, 1: 6, 2: 6}
        assert counts["val"] == {0: 2, 1: 2, 2: 2}
        assert counts["test"] == {0: 2, 1: 2, 2: 2}

    def test_min_sample_per_class_drops_rare_values(
        self, tmp_path, make_cy_parquet
    ):
        rows = _numbered_rows(20)
        rows[-1]["h12"] = 99
        splits = [
            _load_split(
                tmp_path,
                make_cy_parquet,
                rows,
                split_type,
                min_sample_per_class=1,
                label_source="h12",
            )
            for split_type in ("train", "val", "test")
        ]
        seen = sorted(int(data.h11) for ds in splits for data in ds)
        assert seen == list(range(19))

    def test_stratified_needs_populated_classes(
        self, tmp_path, make_cy_parquet
    ):
        # Every `h11` value occurs once, which sklearn cannot stratify.
        with pytest.raises(ValueError, match="least populated"):
            _load_split(
                tmp_path,
                make_cy_parquet,
                _numbered_rows(20),
                "train",
                stratified=True,
                label_source="h11",
            )

    def _names(self, **kwargs):
        obj = CalabiYauDataset.__new__(CalabiYauDataset)
        obj.seed = kwargs.pop("seed", 42)
        obj.split_proportions = kwargs.pop(
            "split_proportions", [0.6, 0.2, 0.2]
        )
        obj.stratified = kwargs.pop("stratified", False)
        obj.label_source = kwargs.pop("label_source", "h11")
        obj.min_sample_per_class = kwargs.pop("min_sample_per_class", None)
        return obj.processed_file_names

    def test_file_names_encode_split_options(self):
        assert self._names() == [
            "train_seed42.pt",
            "val_seed42.pt",
            "test_seed42.pt",
        ]
        assert self._names(
            stratified=True, min_sample_per_class=2, label_source="h11"
        ) == [
            "train_seed42_ccf2_strat_h11.pt",
            "val_seed42_ccf2_strat_h11.pt",
            "test_seed42_ccf2_strat_h11.pt",
        ]
        assert self._names(seed=7, split_proportions=[0.5, 0.1, 0.4]) == [
            "train_seed7_sp0.5-0.1-0.4.pt",
            "val_seed7_sp0.5-0.1-0.4.pt",
            "test_seed7_sp0.5-0.1-0.4.pt",
        ]
