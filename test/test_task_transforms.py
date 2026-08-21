"""Tests for ``mantra.transforms.task_transforms``."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from mantra.manifold_types import Manifold2Type, Manifold3Type
from mantra.transforms.task_transforms import (
    NAME_TO_CLASS_2M,
    NAME_TO_CLASS_3M,
    AttributeToClassTransform,
    AttributeToRegressionTransform,
    BettiToClassTransform,
    NameToClass2MTransform,
    NameToClass3MTransform,
    OrientableToClassTransform,
)


class TestNameToClass2M:
    def test_is_exactly_the_enum_values(self):
        # The map covers every enum value once and nothing else: no dead
        # "" entry and no "#^2 RP^2" alias (the dataset stores the Klein
        # bottle canonically).
        assert set(NAME_TO_CLASS_2M) == set(m.value for m in Manifold2Type)
        assert "" not in NAME_TO_CLASS_2M
        assert "#^2 RP^2" not in NAME_TO_CLASS_2M

    def test_indices_are_contiguous_from_zero(self):
        # 22 enum classes -> indices 0..21 with no gaps.
        assert set(NAME_TO_CLASS_2M.values()) == set(range(22))

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("S^2", 0),
            ("T^2", 1),
            ("RP^2", 9),
            ("Klein bottle", 10),
            ("#^17 RP^2", 21),
        ],
    )
    def test_transform_maps_name_to_class(self, name, expected):
        transform = NameToClass2MTransform()
        result = transform.forward(Data(name=name))
        assert result.y.item() == expected
        assert result.y.dtype == torch.long

    def test_unknown_name_raises(self):
        # Safeguard: an unrecognised label (e.g. the non-canonical
        # "#^2 RP^2", or anything outside the enum) is rejected.
        transform = NameToClass2MTransform()
        for bad in ("#^2 RP^2", "not a manifold"):
            with pytest.raises(KeyError, match="Unknown 2-manifold name"):
                transform.forward(Data(name=bad))

    def test_transform_requires_name(self):
        with pytest.raises(AssertionError):
            NameToClass2MTransform().forward(Data())

    def test_num_classes(self):
        assert NameToClass2MTransform().num_classes == 22


class TestNameToClass3M:
    def test_is_exactly_the_enum_values(self):
        assert set(NAME_TO_CLASS_3M) == set(m.value for m in Manifold3Type)

    def test_indices_are_contiguous_from_zero(self):
        # 9 enum classes -> indices 0..8 with no gaps.
        assert set(NAME_TO_CLASS_3M.values()) == set(range(9))

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("S^3", 0),
            ("S^2 x S^1", 1),
            ("T^3", 6),
            ("(S^2 x S^1)#(S^2 x S^1)", 8),
        ],
    )
    def test_transform_maps_name_to_class(self, name, expected):
        result = NameToClass3MTransform()(Data(name=name))
        assert result.y.item() == expected
        assert result.y.dtype == torch.long

    def test_unknown_name_raises(self):
        # A 2-manifold name is not a valid 3-manifold class.
        with pytest.raises(KeyError, match="Unknown 3-manifold name"):
            NameToClass3MTransform()(Data(name="S^2"))

    def test_num_classes(self):
        assert NameToClass3MTransform().num_classes == 9


class TestAttributeToClassTransform:
    def test_explicit_mapping(self):
        transform = AttributeToClassTransform("kind", mapping={"a": 0, "b": 1})
        assert transform(Data(kind="b")).y.item() == 1
        assert transform.num_classes == 2

    def test_mapping_with_tensor_value(self):
        transform = AttributeToClassTransform("genus", mapping={2: 0, 3: 1})
        assert transform(Data(genus=torch.tensor(3))).y.item() == 1

    def test_float_tensor_raises(self):
        transform = AttributeToClassTransform("genus", mapping={})
        with pytest.raises(AssertionError, match="type int"):
            transform(Data(genus=torch.tensor(1.5)))

    def test_non_scalar_tensor_raises(self):
        transform = AttributeToClassTransform("betti_numbers", mapping={})
        with pytest.raises(AssertionError, match="scalar"):
            transform(Data(betti_numbers=torch.tensor([1, 0, 1])))

    def test_missing_source_raises(self):
        with pytest.raises(AssertionError, match="not present"):
            AttributeToClassTransform(source="genus", mapping={})(Data())

    def test_num_classes_empty_mapping(self):
        assert AttributeToClassTransform("genus", mapping={}).num_classes == 0


class TestOrientableToClassTransform:
    def test_orientable_last_betti_one(self):
        result = OrientableToClassTransform()(Data(betti_numbers=[1, 0, 1]))
        assert result.y.item() == 1

    def test_non_orientable_last_betti_zero(self):
        result = OrientableToClassTransform()(Data(betti_numbers=[1, 0, 0]))
        assert result.y.item() == 0


class TestBettiToClassTransform:
    def test_2d_shape(self):
        result = BettiToClassTransform(manifold_dim=2)(
            Data(betti_numbers=[1, 2, 1])
        )
        assert result.y.shape == (1, 3)

    def test_3d_shape(self):
        result = BettiToClassTransform(manifold_dim=3)(
            Data(betti_numbers=[1, 0, 0, 1])
        )
        assert result.y.shape == (1, 4)

    def test_invalid_dim_raises(self):
        with pytest.raises(AssertionError):
            BettiToClassTransform(manifold_dim=4)


class TestAttributeToRegressionTransform:
    def test_scalar_source(self):
        result = AttributeToRegressionTransform("n_vertices")(
            Data(n_vertices=torch.tensor(10))
        )
        assert result.y.shape == (1, 1)
        assert result.y.item() == 10.0

    def test_same_shape_on_pre_transform_and_transform_path(self):
        # As a `pre_transform` the attribute is a Python int; after the
        # dataset has been collated it is a one-element tensor. Both
        # must yield the same target shape so that batches agree.
        transform = AttributeToRegressionTransform("genus")
        raw = transform(Data(genus=2)).y
        collated = transform(Data(genus=torch.tensor([2]))).y
        assert raw.shape == collated.shape == (1, 1)
        assert raw.dtype == collated.dtype == torch.float32

    def test_vector_attribute(self):
        result = AttributeToRegressionTransform("betti_numbers")(
            Data(betti_numbers=[1, 0, 1])
        )
        assert result.y.shape == (1, 3)
        assert result.y.tolist() == [[1.0, 0.0, 1.0]]

    def test_accepts_numpy_scalars(self):
        result = AttributeToRegressionTransform("value")(
            Data(value=np.float64(2.5))
        )
        assert result.y.item() == 2.5

    def test_missing_source_raises(self):
        transform = AttributeToRegressionTransform("missing")
        with pytest.raises(AttributeError, match="has no attribute"):
            transform(Data(genus=2))
