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
