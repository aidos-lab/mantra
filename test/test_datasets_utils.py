"""Tests for ``mantra.datasets.utils``."""

import pytest

from mantra.datasets import utils


class _FakeResponse:
    def __init__(self, names):
        self._names = names

    def json(self):
        return [{"name": n} for n in self._names]


def test_latest_url_unbalanced():
    url = utils._get_mantra_dataset_url("latest", 2, balanced=False)
    assert url.endswith("/latest/download/2_manifolds.json.gz")


def test_latest_url_balanced():
    url = utils._get_mantra_dataset_url("latest", 3, balanced=True)
    assert url.endswith("/latest/download/3_manifolds_balanced.json.gz")


def test_versioned_url_found(monkeypatch):
    monkeypatch.setattr(
        utils.requests,
        "get",
        lambda *a, **k: _FakeResponse(["v1.0.0", "v2.0.0"]),
    )
    url = utils._get_mantra_dataset_url("v1.0.0", 2, balanced=True)
    assert url.endswith("/download/v1.0.0/2_manifolds_balanced.json.gz")


def test_versioned_url_unknown_raises(monkeypatch):
    monkeypatch.setattr(
        utils.requests, "get", lambda *a, **k: _FakeResponse(["v1.0.0"])
    )
    with pytest.raises(ValueError, match="not available"):
        utils._get_mantra_dataset_url("v9.9.9", 2)
