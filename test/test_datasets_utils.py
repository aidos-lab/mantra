"""Tests for ``mantra.datasets.utils``."""

import os

import pytest

from mantra.datasets import utils


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _releases(tags):
    return _FakeResponse(
        [{"tag_name": t, "name": f"Release {t}"} for t in tags]
    )


@pytest.fixture(autouse=True)
def clear_resolve_cache():
    """The latest-tag resolution is memoized per process."""
    utils._resolve_latest_version.cache_clear()
    yield
    utils._resolve_latest_version.cache_clear()


def test_latest_url():
    url = utils._get_mantra_dataset_url("latest", 2)
    assert url.endswith("/latest/download/2_manifolds.json.gz")


def test_versioned_url_found(monkeypatch):
    monkeypatch.setattr(
        utils.requests,
        "get",
        lambda *a, **k: _releases(["v1.0.0", "v2.0.0"]),
    )
    url = utils._get_mantra_dataset_url("v1.0.0", 2)
    assert url.endswith("/download/v1.0.0/2_manifolds.json.gz")


def test_versioned_url_unknown_raises(monkeypatch):
    monkeypatch.setattr(
        utils.requests, "get", lambda *a, **k: _releases(["v1.0.0"])
    )
    with pytest.raises(ValueError, match="not available"):
        utils._get_mantra_dataset_url("v9.9.9", 2)


def test_versioned_url_matches_tag_name_over_title(monkeypatch):
    # Releases whose display name differs from the tag must validate by tag.
    monkeypatch.setattr(
        utils.requests,
        "get",
        lambda *a, **k: _FakeResponse([{"tag_name": "v1.0.0", "name": "Big"}]),
    )
    url = utils._get_mantra_dataset_url("v1.0.0", 3)
    assert url.endswith("/download/v1.0.0/3_manifolds.json.gz")


def test_versioned_url_skips_validation_when_disabled(monkeypatch):
    def fail(*a, **k):  # pragma: no cover
        raise AssertionError("no network calls with validate=False")

    monkeypatch.setattr(utils.requests, "get", fail)
    url = utils._get_mantra_dataset_url("v1.0.0", 2, validate=False)
    assert url.endswith("/download/v1.0.0/2_manifolds.json.gz")


def test_versioned_url_rate_limit_degrades_to_warning(monkeypatch):
    # GitHub returns a dict, not a list, when rate-limited.
    monkeypatch.setattr(
        utils.requests,
        "get",
        lambda *a, **k: _FakeResponse({"message": "API rate limit exceeded"}),
    )
    with pytest.warns(UserWarning, match="rate limit"):
        url = utils._get_mantra_dataset_url("v1.0.0", 2)
    assert url.endswith("/download/v1.0.0/2_manifolds.json.gz")


def test_versioned_url_network_error_degrades_to_warning(monkeypatch):
    def raise_error(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setattr(utils.requests, "get", raise_error)
    with pytest.warns(UserWarning, match="Could not validate"):
        url = utils._get_mantra_dataset_url("v1.0.0", 2)
    assert url.endswith("/download/v1.0.0/2_manifolds.json.gz")


class _FakeHeadResponse:
    def __init__(self, location):
        self.headers = {"Location": location}


def test_resolve_latest_version(monkeypatch):
    monkeypatch.setattr(
        utils.requests,
        "head",
        lambda *a, **k: _FakeHeadResponse(
            "https://github.com/aidos-lab/MANTRA/releases/tag/v0.0.19"
        ),
    )
    assert utils._resolve_latest_version() == "v0.0.19"


def test_resolve_latest_version_is_memoized(monkeypatch):
    calls = []

    def head(*a, **k):
        calls.append(1)
        return _FakeHeadResponse(".../tag/v0.0.19")

    monkeypatch.setattr(utils.requests, "head", head)
    assert utils._resolve_latest_version() == "v0.0.19"
    assert utils._resolve_latest_version() == "v0.0.19"
    assert len(calls) == 1


def test_resolve_latest_version_falls_back_offline(monkeypatch):
    def raise_error(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setattr(utils.requests, "head", raise_error)
    with pytest.warns(UserWarning, match="Could not resolve"):
        assert utils._resolve_latest_version() == "latest"


class TestFindCachedVersion:
    def test_picks_newest_cached_release(self, tmp_path):
        for tag in ["v0.0.9", "v0.0.19", "v0.0.2"]:
            os.makedirs(tmp_path / "mantra" / tag / "2D")
        assert utils._find_cached_version(str(tmp_path), 2) == "v0.0.19"

    def test_ignores_other_dimensions_and_junk(self, tmp_path):
        os.makedirs(tmp_path / "mantra" / "v0.0.9" / "3D")
        os.makedirs(tmp_path / "mantra" / "not-a-version" / "2D")
        assert utils._find_cached_version(str(tmp_path), 2) is None

    def test_no_cache_directory(self, tmp_path):
        assert utils._find_cached_version(str(tmp_path), 2) is None


def test_filter_by_class_count_none_disables_filtering():
    entries = [{"name": "S^2"}, {"name": "RP^2"}]
    filtered, counts = utils.filter_by_class_count(entries, "name", None)
    assert filtered == entries
    assert not counts


def test_filter_by_class_count_drops_rare_classes():
    entries = [{"name": "S^2"}] * 3 + [{"name": "RP^2"}]
    filtered, counts = utils.filter_by_class_count(entries, "name", 1)
    assert all(e["name"] == "S^2" for e in filtered)
    assert len(filtered) == 3
    assert counts == {"S^2": 3, "RP^2": 1}
