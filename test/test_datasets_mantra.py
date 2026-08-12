"""Tests for ``mantra.datasets.mantra`` (and the abstract base)."""

import json
import os
from collections import Counter

import pytest

import mantra.augmentations.balancing as balancing_mod
import mantra.datasets.mantra as mantra_mod
from mantra.datasets import ManifoldTriangulations

from .conftest import manifold_entry


def test_local_path_loads_dataset(
    make_manifolds_json, balanced_entries, tmp_path
):
    path = make_manifolds_json(balanced_entries)
    ds = ManifoldTriangulations(
        str(tmp_path / "root"), dimension=2, local_path=path
    )
    assert len(ds) == len(balanced_entries)
    assert ds[0].name == "S^2"
    assert ds.raw_file_names == ["2_manifolds.json"]
    assert ds.processed_file_names == ["full.pt"]


def test_balanced_suffix(
    make_manifolds_json, balanced_entries, tmp_path, no_dedup
):
    path = make_manifolds_json(balanced_entries)
    ds = ManifoldTriangulations(
        str(tmp_path / "root"),
        dimension=2,
        balanced=True,
        local_path=path,
        target_count=2,
        n_moves=1,
        use_surgery=False,
    )
    assert ds.raw_file_names == ["2_manifolds.json"]
    assert ds.processed_file_names == ["full.pt"]
    assert os.path.basename(ds.processed_dir) == (
        "balanced_42_n_moves1_target_count2_use_surgeryFalse"
    )


def test_invalid_dimension_raises(tmp_path):
    with pytest.raises(AssertionError):
        ManifoldTriangulations(str(tmp_path / "root"), dimension=5)


def test_name_changes_processed_dir(
    make_manifolds_json, balanced_entries, tmp_path
):
    path = make_manifolds_json(balanced_entries)
    ds = ManifoldTriangulations(
        str(tmp_path / "root"), dimension=2, name="custom", local_path=path
    )
    assert os.path.join("processed", "custom") in ds.processed_dir


def test_pre_filter_and_pre_transform_applied(
    make_manifolds_json, balanced_entries, tmp_path
):
    path = make_manifolds_json(balanced_entries)

    def pre_filter(d):
        return d.orientable

    def pre_transform(d):
        d.tagged = True
        return d

    ds = ManifoldTriangulations(
        str(tmp_path / "root"),
        dimension=2,
        local_path=path,
        pre_filter=pre_filter,
        pre_transform=pre_transform,
    )
    assert len(ds) == 5
    assert all(d.tagged for d in ds)


def test_url_download_branch(
    make_manifolds_json, balanced_entries, tmp_path, monkeypatch
):
    content = json.dumps(balanced_entries)

    def fake_download_url(url, raw_dir):
        gz = os.path.join(raw_dir, "2_manifolds.json.gz")
        with open(gz, "w") as f:
            f.write("dummy gz payload")
        fake_download_url.url = url
        return gz

    def fake_extract_gz(path, raw_dir):
        with open(os.path.join(raw_dir, "2_manifolds.json"), "w") as f:
            f.write(content)

    monkeypatch.setattr(mantra_mod, "download_url", fake_download_url)
    monkeypatch.setattr(mantra_mod, "extract_gz", fake_extract_gz)

    ds = ManifoldTriangulations(
        str(tmp_path / "root"), dimension=2, version="latest"
    )
    assert len(ds) == len(balanced_entries)
    assert fake_download_url.url.endswith("2_manifolds.json.gz")


def test_add_version_to_root_branches():
    obj = ManifoldTriangulations.__new__(ManifoldTriangulations)
    obj.dimension = 2
    obj.version = "v1.0.0"
    assert obj._add_version_to_root() == "/mantra/v1.0.0/2D"
    obj.version = "latest"
    assert obj._add_version_to_root() == "/mantra/2D"


class TestBalancedProcessing:
    def test_balance_dataset_called_with_seed_and_kwargs(
        self, make_manifolds_json, balanced_entries, tmp_path, monkeypatch
    ):
        calls = {}

        def spy(dataset, **kwargs):
            calls["n_entries"] = len(dataset)
            calls["kwargs"] = kwargs
            return dataset[:4]

        monkeypatch.setattr(mantra_mod, "balance_dataset", spy)
        path = make_manifolds_json(balanced_entries)
        ds = ManifoldTriangulations(
            str(tmp_path / "root"),
            dimension=2,
            balanced=True,
            local_path=path,
            seed=7,
            max_vertices=9,
            target_count=3,
            n_moves=2,
        )
        assert calls["n_entries"] == len(balanced_entries)
        assert calls["kwargs"] == dict(
            seed=7, max_vertices=9, target_count=3, n_moves=2, use_surgery=True
        )
        assert len(ds) == 4

    def test_unbalanced_skips_balance_dataset(
        self, make_manifolds_json, balanced_entries, tmp_path, monkeypatch
    ):
        def fail(*a, **k):  # pragma: no cover
            raise AssertionError("balance_dataset must not be called")

        monkeypatch.setattr(mantra_mod, "balance_dataset", fail)
        path = make_manifolds_json(balanced_entries)
        ds = ManifoldTriangulations(
            str(tmp_path / "root"), dimension=2, local_path=path
        )
        assert len(ds) == len(balanced_entries)

    def test_balancing_equalizes_class_counts(
        self, make_manifolds_json, tmp_path, no_dedup
    ):
        # Imbalanced input: oversampling has to top up RP^2.
        entries = [manifold_entry(f"s{i}", name="S^2") for i in range(6)] + [
            manifold_entry(f"r{i}", name="RP^2", orientable=False)
            for i in range(2)
        ]
        path = make_manifolds_json(entries)
        ds = ManifoldTriangulations(
            str(tmp_path / "root"),
            dimension=2,
            balanced=True,
            local_path=path,
            target_count=3,
            n_moves=1,
            use_surgery=False,
        )
        counts = Counter(d.name for d in ds)
        assert counts == {"S^2": 3, "RP^2": 3}
        ids = [d.id for d in ds]
        assert len(ids) == len(set(ids))

    def test_topology_changes_create_missing_classes(
        self, make_manifolds_json, balanced_entries, tmp_path, no_dedup
    ):
        path = make_manifolds_json(balanced_entries)
        ds = ManifoldTriangulations(
            str(tmp_path / "root"),
            dimension=2,
            balanced=True,
            local_path=path,
            target_count=2,
            n_moves=1,
            use_surgery=True,
        )
        counts = Counter(d.name for d in ds)
        # Gluing reaches classes absent from the input.
        assert "T^2" in counts
        assert "Klein bottle" in counts
        assert all(c == 2 for c in counts.values())

    def test_deduplication_runs_during_balancing(
        self, make_manifolds_json, balanced_entries, tmp_path, monkeypatch
    ):
        scanned = []

        def spy(entries, verbose=False):
            scanned.append({e["name"] for e in entries})
            return []

        monkeypatch.setattr(balancing_mod, "find_duplicates", spy)
        # RP^2 (2 entries) needs augmentation; S^2 (6 entries) does not.
        entries = [manifold_entry(f"s{i}", name="S^2") for i in range(6)] + [
            manifold_entry(f"r{i}", name="RP^2", orientable=False)
            for i in range(2)
        ]
        path = make_manifolds_json(entries)
        ManifoldTriangulations(
            str(tmp_path / "root"),
            dimension=2,
            balanced=True,
            local_path=path,
            target_count=3,
            n_moves=1,
            use_surgery=False,
        )
        # Dedup ran on the augmented class only; the untouched class is
        # skipped to avoid pointless isomorphism scans.
        assert scanned == [{"RP^2"}]

    def test_balancing_is_deterministic(
        self, make_manifolds_json, balanced_entries, tmp_path, no_dedup
    ):
        kwargs = dict(
            dimension=2,
            balanced=True,
            seed=3,
            target_count=4,
            n_moves=2,
            use_surgery=False,
        )
        path = make_manifolds_json(balanced_entries)
        ids = [
            [
                d.id
                for d in ManifoldTriangulations(
                    str(tmp_path / root), local_path=path, **kwargs
                )
            ]
            for root in ["root_a", "root_b"]
        ]
        assert ids[0] == ids[1]
