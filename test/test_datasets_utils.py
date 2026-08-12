"""Tests for ``mantra.datasets.utils``."""

import pytest

from mantra.datasets import utils


class _FakeResponse:
    def __init__(self, names):
        self._names = names

    def json(self):
        return [{"name": n} for n in self._names]


def test_latest_url():
    url = utils._get_mantra_dataset_url("latest", 2)
    assert url.endswith("/latest/download/2_manifolds.json.gz")


def test_versioned_url_found(monkeypatch):
    monkeypatch.setattr(
        utils.requests,
        "get",
        lambda *a, **k: _FakeResponse(["v1.0.0", "v2.0.0"]),
    )
    url = utils._get_mantra_dataset_url("v1.0.0", 2)
    assert url.endswith("/download/v1.0.0/2_manifolds.json.gz")


def test_versioned_url_unknown_raises(monkeypatch):
    monkeypatch.setattr(
        utils.requests, "get", lambda *a, **k: _FakeResponse(["v1.0.0"])
    )
    with pytest.raises(ValueError, match="not available"):
        utils._get_mantra_dataset_url("v9.9.9", 2)


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
