"""Tests for ``mantra.datasets.mantra`` (and the abstract base)."""

import json
import os

import pytest

import mantra.datasets.mantra as mantra_mod
from mantra.datasets import ManifoldTriangulations


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


def test_balanced_suffix(make_manifolds_json, balanced_entries, tmp_path):
    path = make_manifolds_json(balanced_entries)
    ds = ManifoldTriangulations(
        str(tmp_path / "root"), dimension=2, balanced=True, local_path=path
    )
    assert ds.raw_file_names == ["2_manifolds.json"]
    assert ds.processed_file_names == ["full.pt"]


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
    monkeypatch.setattr(
        mantra_mod, "_resolve_latest_version", lambda: "latest"
    )

    ds = ManifoldTriangulations(
        str(tmp_path / "root"), dimension=2, version="latest"
    )
    assert len(ds) == len(balanced_entries)
    assert fake_download_url.url.endswith("2_manifolds.json.gz")


def test_latest_resolves_to_versioned_root(
    make_manifolds_json, balanced_entries, tmp_path, monkeypatch
):
    content = json.dumps(balanced_entries)

    def fake_download_url(url, raw_dir):
        gz = os.path.join(raw_dir, "2_manifolds.json.gz")
        with open(gz, "w") as f:
            f.write("dummy gz payload")
        return gz

    def fake_extract_gz(path, raw_dir):
        with open(os.path.join(raw_dir, "2_manifolds.json"), "w") as f:
            f.write(content)

    monkeypatch.setattr(mantra_mod, "download_url", fake_download_url)
    monkeypatch.setattr(mantra_mod, "extract_gz", fake_extract_gz)
    monkeypatch.setattr(
        mantra_mod, "_resolve_latest_version", lambda: "v1.2.3"
    )
    url_calls = {}

    def fake_url(version, dimension, balanced=False, validate=True):
        url_calls["validate"] = validate
        return f"https://example.org/{version}/x.json.gz"

    monkeypatch.setattr(mantra_mod, "_get_mantra_dataset_url", fake_url)

    ds = ManifoldTriangulations(
        str(tmp_path / "root"), dimension=2, version="latest"
    )
    assert ds.version == "v1.2.3"
    assert os.path.join("mantra", "v1.2.3", "2D") in ds.root
    # Tags resolved from `latest` come from GitHub itself and must not
    # trigger the release-listing validation round-trip.
    assert url_calls["validate"] is False


def test_offline_fallback_uses_cached_version(
    make_manifolds_json, balanced_entries, tmp_path, monkeypatch
):
    # A previous online run left a cached release on disk.
    content = json.dumps(balanced_entries)
    raw_dir = tmp_path / "root" / "mantra" / "v0.0.9" / "2D" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "2_manifolds.json").write_text(content)

    # Now the network is gone: resolution falls back to "latest".
    monkeypatch.setattr(
        mantra_mod, "_resolve_latest_version", lambda: "latest"
    )

    with pytest.warns(UserWarning, match="cached MANTRA release v0.0.9"):
        ds = ManifoldTriangulations(
            str(tmp_path / "root"), dimension=2, version="latest"
        )
    assert ds.version == "v0.0.9"
    assert len(ds) == len(balanced_entries)


def test_local_path_skips_latest_resolution(
    make_manifolds_json, balanced_entries, tmp_path, monkeypatch
):
    def fail(*a, **k):  # pragma: no cover
        raise AssertionError("resolver must not be called with local_path")

    monkeypatch.setattr(mantra_mod, "_resolve_latest_version", fail)
    path = make_manifolds_json(balanced_entries)
    ds = ManifoldTriangulations(
        str(tmp_path / "root"), dimension=2, local_path=path
    )
    assert ds.version == "latest"


def test_add_version_to_root_branches():
    obj = ManifoldTriangulations.__new__(ManifoldTriangulations)
    obj.dimension = 2
    obj.version = "v1.0.0"
    assert obj._add_version_to_root() == "/mantra/v1.0.0/2D"
    obj.version = "latest"
    assert obj._add_version_to_root() == "/mantra/2D"
