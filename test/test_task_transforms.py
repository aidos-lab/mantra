"""Regression tests for ``mantra.transforms.task_transforms``.

Covers:
- ``NameToClass2MTransform`` is callable (BaseTransform inheritance).
- ``NAME_TO_CLASS_2M`` covers every name in the released datasets, has
  no index holes, and aliases ``Klein bottle`` to ``#^2 RP^2``.
- ``data.y`` from both ``NameToClass2MTransform`` and
  ``OrientableToClassTransform`` is 1-d, so PyG's ``num_classes`` works.
"""

import pytest
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import Compose

from mantra.manifold_types import Manifold3Type
from mantra.transforms.task_transforms import (
    NAME_TO_CLASS_2M,
    NAME_TO_CLASS_3M,
    BettiToClassTransform,
    BinaryHomeomorphicTransform,
    NameToClass2MTransform,
    NameToClass3MTransform,
    OrientableToClassTransform,
)


# Every name observed in the released 2-manifolds datasets
# (`2_manifolds.json` and `2_manifolds_balanced.json` at the time of this
# test). Enumerated by reading both JSON files directly.
RELEASED_NAMES_2M = {
    "S^2",
    "T^2",
    "#^2 T^2",
    "#^3 T^2",
    "#^4 T^2",
    "#^5 T^2",
    "#^6 T^2",
    "#^7 T^2",
    "#^8 T^2",
    "RP^2",
    "Klein bottle",
    "#^3 RP^2",
    "#^4 RP^2",
    "#^5 RP^2",
    "#^6 RP^2",
    "#^7 RP^2",
    "#^8 RP^2",
    "#^10 RP^2",
    "#^12 RP^2",
    "#^15 RP^2",
    "#^16 RP^2",
    "#^17 RP^2",
}


# --- F2: callable via Compose -------------------------------------------------

def test_name_transform_is_callable(mk_data):
    # The pre-fix failure mode is TypeError because the class wasn't
    # callable; we only need to assert that this no longer raises and
    # that y is populated. (BaseTransform.__call__ copies the data.)
    data = mk_data(name="T^2")
    out = NameToClass2MTransform()(data)
    assert "y" in out


def test_name_transform_via_compose(mk_data):
    data = mk_data(name="T^2")
    out = Compose([NameToClass2MTransform()])(data)
    assert "y" in out


# --- F3: NAME_TO_CLASS_2M is complete, dense, and aliases correctly -----------

def test_name_to_class_covers_released_datasets():
    missing = RELEASED_NAMES_2M - set(NAME_TO_CLASS_2M)
    assert not missing, f"missing keys: {sorted(missing)}"


def test_klein_bottle_alias():
    assert NAME_TO_CLASS_2M["Klein bottle"] == NAME_TO_CLASS_2M["#^2 RP^2"]


def test_name_to_class_indices_dense():
    values = sorted(set(NAME_TO_CLASS_2M.values()))
    assert values == list(range(max(values) + 1)), (
        f"class indices have holes: {values}"
    )


def test_name_to_class_indices_non_negative():
    assert all(v >= 0 for v in NAME_TO_CLASS_2M.values())


# --- 3-manifold name -> class mapping ----------------------------------------

# Every name observed in the released 3-manifolds datasets
# (`3_manifolds.json` and `3_manifolds_balanced.json`).
RELEASED_NAMES_3M = {
    "S^3",
    "S^2 x S^1",
    "S^2 twist S^1",
    "RP^3",
    "L(3,1)",
    "L(4,1)",
    "T^3",
    "S^3/Q",
    "(S^2 x S^1)#(S^2 x S^1)",
}


def test_name_to_class_3m_covers_released_datasets():
    missing = RELEASED_NAMES_3M - set(NAME_TO_CLASS_3M)
    assert not missing, f"missing keys: {sorted(missing)}"


def test_name_to_class_3m_matches_enum():
    # The mapping must stay in sync with the canonical type enum.
    enum_names = {t.value for t in Manifold3Type}
    assert set(NAME_TO_CLASS_3M) == enum_names


def test_name_to_class_3m_indices_dense():
    values = sorted(set(NAME_TO_CLASS_3M.values()))
    assert values == list(range(max(values) + 1)), (
        f"class indices have holes: {values}"
    )


def test_name_to_class_3m_indices_unique():
    # 3-manifold types are all distinct (no aliasing like Klein/#^2 RP^2).
    values = list(NAME_TO_CLASS_3M.values())
    assert len(values) == len(set(values))


def test_name_transform_3m_y(mk_data):
    data = mk_data(name="RP^3")
    out = NameToClass3MTransform()(data)
    assert out.y.tolist() == [NAME_TO_CLASS_3M["RP^3"]]
    assert out.y.dtype == torch.long


def test_name_transform_3m_via_compose(mk_data):
    data = mk_data(name="T^3")
    out = Compose([NameToClass3MTransform()])(data)
    assert "y" in out


# --- F4: data.y is 1-d after both transforms ---------------------------------

def test_name_transform_y_is_1d(mk_data):
    data = mk_data(name="T^2")
    out = NameToClass2MTransform()(data)
    assert out.y.ndim >= 1
    assert out.y.tolist() == [NAME_TO_CLASS_2M["T^2"]]


def test_orientable_transform_y_is_1d(mk_data):
    data = mk_data(betti_numbers=[1, 2, 1])
    out = OrientableToClassTransform()(data)
    assert out.y.ndim >= 1
    assert out.y.tolist() == [1]


def test_orientable_transform_non_orientable(mk_data):
    # Non-orientable surfaces have a vanishing top Betti number (b_2 = 0),
    # which is the value the transform reads off as the class label.
    data = mk_data(betti_numbers=[1, 1, 0])
    out = OrientableToClassTransform()(data)
    assert out.y.tolist() == [0]
    assert out.y.dtype == torch.long


# --- BettiToClassTransform ----------------------------------------------------

def test_betti_transform_2manifold_shape(mk_data):
    data = mk_data(betti_numbers=[1, 0, 1])
    out = BettiToClassTransform(manifold_dim=2)(data)
    assert out.y.shape == (1, 3)
    assert out.y.dtype == torch.float
    assert out.y.tolist() == [[1.0, 0.0, 1.0]]


def test_betti_transform_3manifold_shape(mk_data):
    data = mk_data(betti_numbers=[1, 0, 0, 1])
    out = BettiToClassTransform(manifold_dim=3)(data)
    assert out.y.shape == (1, 4)


def test_betti_transform_rejects_bad_dim():
    with pytest.raises(AssertionError):
        BettiToClassTransform(manifold_dim=4)


# --- BinaryHomeomorphicTransform ---------------------------------------------

def test_binary_homeomorphic_same(mk_data):
    data = mk_data(
        triangulation=torch.zeros(2, 4, 3), name=["S^2", "S^2"]
    )
    out = BinaryHomeomorphicTransform()(data)
    assert out.y.tolist() == [1]
    assert out.y.dtype == torch.long


def test_binary_homeomorphic_different(mk_data):
    data = mk_data(
        triangulation=torch.zeros(2, 4, 3), name=["S^2", "T^2"]
    )
    out = BinaryHomeomorphicTransform()(data)
    assert out.y.tolist() == [0]


def test_binary_homeomorphic_requires_pair(mk_data):
    # A single (non-paired) triangulation must be rejected.
    data = mk_data(triangulation=torch.zeros(1, 4, 3), name=["S^2"])
    with pytest.raises(AssertionError):
        BinaryHomeomorphicTransform()(data)


# --- Integration: PyG ``num_classes`` works through the full pipeline --------

class _TinyDataset(InMemoryDataset):
    """In-memory PyG dataset built directly from a list of ``Data`` objects.

    Used here purely to drive ``num_classes`` through the same code path as
    the user's failing example.
    """

    def __init__(self, data_list, transform=None):
        super().__init__(root=None, transform=transform)
        self.data, self.slices = self.collate(data_list)


def test_num_classes_via_pyg(mk_data):
    samples = [
        mk_data(name=n)
        for n in ["S^2", "T^2", "Klein bottle", "#^3 RP^2"]
    ]
    ds = _TinyDataset(samples, transform=NameToClass2MTransform())
    # max class index + 1 over the 4 samples we provided
    expected = (
        max(NAME_TO_CLASS_2M[n] for n in {s.name for s in samples}) + 1
    )
    assert ds.num_classes == expected


# --- Opt-in full-dataset check (skips without --dataset-path) ----------------

def test_every_dataset_name_is_in_dict(dataset):
    seen = {d["name"] for d in dataset}
    # Pick the right mapping from the dataset's manifold dimension so the
    # check works for both 2- and 3-manifold dataset files.
    dim = next(iter(dataset)).get("dimension")
    mapping = NAME_TO_CLASS_3M if dim == 3 else NAME_TO_CLASS_2M
    missing = seen - set(mapping)
    assert not missing, (
        f"name->class mapping missing {len(missing)} names from the "
        f"dataset: {sorted(missing)}"
    )
