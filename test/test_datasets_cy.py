"""Tests for the parquet-based CY dataset and its splits."""

from collections import Counter

import pytest
import torch
from torch_geometric.transforms import BaseTransform

from mantra.datasets import CalabiYau, CalabiYauDataset


class _CountSimplices(BaseTransform):
    def forward(self, data):
        data.n_simplices = len(data.triangulation)
        data.coords = data.coords * 2
        return data


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

        # Simplices are converted to the 1-indexed MANTRA convention
        # (sorted within and across simplices), `dimension` holds the
        # topological dimension.
        assert data.triangulation == [
            [1, 2, 3],
            [1, 2, 5],
            [1, 3, 4],
            [1, 4, 5],
        ]
        assert int(data.dimension) == 2
        assert int(data.n_vertices) == 5

        assert data.coords.dtype == torch.float32
        assert data.coords.shape == (5, 2)

        # Extra parquet columns become attributes.
        assert int(data.h11) == 6
        assert int(data.h12) == 46
        assert int(dataset[1].h11) == 7

    def test_parallel_pre_transform_matches_serial(
        self, tmp_path, make_cy_parquet, cy_rows
    ):
        rows = cy_rows * 8
        serial = CalabiYau(
            root=str(tmp_path / "serial"),
            local_path=make_cy_parquet(rows, "serial.parquet"),
            pre_transform=_CountSimplices(),
            pre_transform_workers=1,
        )
        parallel = CalabiYau(
            root=str(tmp_path / "parallel"),
            local_path=make_cy_parquet(rows, "parallel.parquet"),
            pre_transform=_CountSimplices(),
            pre_transform_workers=2,
        )

        assert len(parallel) == len(serial) == len(rows)
        for a, b in zip(serial, parallel):
            assert a.n_simplices == b.n_simplices == 4
            assert torch.equal(a.coords, b.coords)
            assert int(a.h11) == int(b.h11)

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


class TestCYClassFilter:
    def test_min_sample_per_class_drops_rare_values(
        self, tmp_path, make_cy_parquet
    ):
        # h12 attains 0, 1, 2 ten times each and 7 twice.
        rows = _numbered_rows(30)
        for i in (30, 31):
            rows.append({**rows[0], "h11": i, "h12": 7})

        splits = [
            _load_split(
                tmp_path,
                make_cy_parquet,
                rows,
                split_type,
                label_source="h12",
                min_sample_per_class=2,
            )
            for split_type in ("train", "val", "test")
        ]

        assert sum(len(split) for split in splits) == 30
        assert all(int(data.h12) != 7 for split in splits for data in split)
        assert "ccf2-h12" in splits[0].processed_paths[0]

        unfiltered = _load_split(tmp_path, make_cy_parquet, rows, "train")
        assert "ccf" not in unfiltered.processed_paths[0]

    def test_filter_on_another_label_source_gets_its_own_split_file(
        self, tmp_path, make_cy_parquet
    ):
        # h11 alternates 0/1 (16 each), h12 attains 7 only twice.
        rows = [
            {**row, "h11": k % 2} for k, row in enumerate(_numbered_rows(30))
        ]
        for i in (0, 1):
            rows.append({**rows[0], "h11": i, "h12": 7})

        by_h12 = _load_split(
            tmp_path,
            make_cy_parquet,
            rows,
            "train",
            label_source="h12",
            min_sample_per_class=2,
        )
        by_h11 = _load_split(
            tmp_path,
            make_cy_parquet,
            rows,
            "train",
            label_source="h11",
            min_sample_per_class=2,
        )

        assert by_h12.processed_paths[0] != by_h11.processed_paths[0]
        assert all(int(data.h12) != 7 for data in by_h12)
        assert any(int(data.h12) == 7 for data in by_h11)

    def test_filter_dropping_everything_is_an_error(
        self, tmp_path, make_cy_parquet
    ):
        with pytest.raises(ValueError, match="drops all 30 samples"):
            _load_split(
                tmp_path,
                make_cy_parquet,
                _numbered_rows(30),
                "train",
                label_source="h12",
                min_sample_per_class=10,
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
