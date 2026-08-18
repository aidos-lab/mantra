"""Tests for ``mantra.transforms.create_labels``."""

import pytest
import torch
from torch_geometric.data import Data

from mantra.transforms.create_labels import CreateLabels


def test_bool_labels_map_to_zero_and_one():
    t = CreateLabels("orientable")
    d_false = t(Data(orientable=False))
    d_true = t(Data(orientable=True))
    assert d_false.y.tolist() == [0]
    assert d_true.y.tolist() == [1]
    assert d_false.label is False
    assert d_true.label is True


def test_string_labels_indexed_in_order_of_appearance():
    t = CreateLabels("name")
    ys = [t(Data(name=n)).y.item() for n in ["S^2", "T^2", "S^2", "RP^2"]]
    assert ys == [0, 1, 0, 2]
    assert t.label_to_index == {"S^2": 0, "T^2": 1, "RP^2": 2}


def test_tensor_label_is_converted_to_scalar():
    t = CreateLabels("genus")
    d = t(Data(genus=torch.tensor(5)))
    assert d.y.item() == 0
    assert t.label_to_index == {5: 0}


def test_remap_path_for_preprocessed_data():
    t = CreateLabels("name")
    ys = [t(Data(label="x", y=torch.tensor([v]))).y.item() for v in [7, 7, 3]]
    assert ys == [0, 0, 1]
    assert t.index_remap == {7: 0, 3: 1}


def test_missing_source_attribute_raises():
    t = CreateLabels("name")
    with pytest.raises(AssertionError, match="not present"):
        t(Data(orientable=True))


def test_multi_source_tuple_labels_indexed_in_order_of_appearance():
    t = CreateLabels(["name", "genus"])
    samples = [("S^2", 0), ("T^2", 1), ("S^2", 0), ("S^2", 1)]
    ys = [t(Data(name=n, genus=g)).y.item() for n, g in samples]

    # One class per distinct value *combination*, indexed compactly in
    # first-appearance order.
    assert ys == [0, 1, 0, 2]
    assert t.label_to_index == {
        ("S^2", 0): 0,
        ("T^2", 1): 1,
        ("S^2", 1): 2,
    }
    assert t(Data(name="T^2", genus=1)).label == ("T^2", 1)


def test_omegaconf_listconfig_source_behaves_like_list():
    from omegaconf import OmegaConf

    t = CreateLabels(OmegaConf.create(["name", "genus"]))

    # The list-like config is coerced into a plain list of plain strings.
    assert t.source == ["name", "genus"]
    assert all(type(s) is str for s in t.source)

    samples = [("S^2", 0), ("T^2", 1), ("S^2", 0), ("S^2", 1)]
    ys = [t(Data(name=n, genus=g)).y.item() for n, g in samples]
    assert ys == [0, 1, 0, 2]
    assert t.label_to_index == {
        ("S^2", 0): 0,
        ("T^2", 1): 1,
        ("S^2", 1): 2,
    }


def test_source_presence_wins_over_stale_precomputed_y():
    t = CreateLabels(["name", "genus"])
    t(Data(name="S^2", genus=0))

    # A sample carrying both the source attributes *and* a stale ``y``
    # is recomputed from the sources; the stale index never enters the
    # remap table.
    d = t(Data(name="T^2", genus=1, y=torch.tensor([99])))
    assert d.y.item() == 1
    assert d.label == ("T^2", 1)
    assert 99 not in t.index_remap


def test_multi_source_only_y_takes_remap_fallback():
    t = CreateLabels(["name", "genus"])
    ys = [t(Data(y=torch.tensor([v]))).y.item() for v in [7, 7, 3]]

    # Without the source attributes the existing indices are remapped
    # compactly, exactly like the single-source fallback.
    assert ys == [0, 0, 1]
    assert t.index_remap == {7: 0, 3: 1}
