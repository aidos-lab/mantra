"""Tests for ``mantra.datasets.homeo_pred``."""

import pytest

import mantra.datasets.homeo_pred as homeo_mod
from mantra.configs import SplitConfig
from mantra.datasets import MANTRA
from mantra.datasets.homeo_pred import PropertyPredictionDS
from mantra.tasks.task_types import TaskType
from mantra.transforms.create_labels import CreateLabels


@pytest.fixture
def patch_mantra(monkeypatch, make_manifolds_json, balanced_entries):
    """Patch the inner MANTRA so it loads from a local fixture (no network)."""
    path = make_manifolds_json(balanced_entries)

    def fake_mantra(root, **kwargs):
        kwargs.setdefault("local_path", path)
        return MANTRA(root, **kwargs)

    monkeypatch.setattr(homeo_mod, "MANTRA", fake_mantra)
    return path


def test_stratified_split_sizes(patch_mantra, tmp_path):
    cfg = SplitConfig(split=(0.6, 0.2, 0.2), seed=0, use_stratified=True)
    root = str(tmp_path / "root")
    kwargs = dict(
        task_type=TaskType.ORIENTABILITY,
        split_config=cfg,
        manifold="2",
        transform=CreateLabels("orientable"),
    )
    train = PropertyPredictionDS(root, split="train", **kwargs)
    val = PropertyPredictionDS(root, split="val", **kwargs)
    test = PropertyPredictionDS(root, split="test", **kwargs)
    assert (len(train), len(val), len(test)) == (6, 2, 2)


def test_unstratified_with_pre_transform_and_no_transform(
    monkeypatch, make_manifolds_json, balanced_entries, tmp_path
):
    path = make_manifolds_json(balanced_entries)

    def fake_mantra(root, **kwargs):
        kwargs.setdefault("local_path", path)
        return MANTRA(root, **kwargs)

    monkeypatch.setattr(homeo_mod, "MANTRA", fake_mantra)

    cfg = SplitConfig(split=(0.6, 0.2, 0.2), seed=1, use_stratified=False)
    # Labels come from the raw dataset's pre_transform; the wrapper applies
    # no transform of its own, exercising the ``transform is None`` branch.
    ds = PropertyPredictionDS(
        str(tmp_path / "root"),
        task_type=TaskType.NAME,
        split="test",
        split_config=cfg,
        manifold="2",
        transform=None,
        pre_transform=CreateLabels("orientable"),
    )
    assert len(ds) == 2


def test_manifold_3_not_implemented(patch_mantra, tmp_path):
    cfg = SplitConfig(split=(0.6, 0.2, 0.2), seed=0, use_stratified=False)
    with pytest.raises(NotImplementedError):
        PropertyPredictionDS(
            str(tmp_path / "root"),
            task_type=TaskType.NAME,
            split="train",
            split_config=cfg,
            manifold="3",
            transform=CreateLabels("orientable"),
        )


def test_unknown_split_raises_value_error(patch_mantra, tmp_path):
    cfg = SplitConfig(split=(0.6, 0.2, 0.2), seed=0, use_stratified=True)
    ds = PropertyPredictionDS(
        str(tmp_path / "root"),
        task_type=TaskType.ORIENTABILITY,
        split="train",
        split_config=cfg,
        manifold="2",
        transform=CreateLabels("orientable"),
    )
    with pytest.raises(ValueError, match="Can not find"):
        ds._get_processed_path(TaskType.ORIENTABILITY, "bogus")


def test_raw_file_names_download_and_filename(patch_mantra, tmp_path):
    cfg = SplitConfig(split=(0.6, 0.2, 0.2), seed=0, use_stratified=True)
    ds = PropertyPredictionDS(
        str(tmp_path / "root"),
        task_type=TaskType.ORIENTABILITY,
        split="train",
        split_config=cfg,
        manifold="2",
        transform=CreateLabels("orientable"),
    )
    assert ds.raw_file_names == []
    assert ds.download() is None
    assert ds._data_filename(TaskType.NAME, "train") == "data_NAME_train.pt"
