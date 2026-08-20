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
    @pytest.mark.parametrize(
        "value,expected",
        [
            (7, 7),
            (True, 1),
            (False, 0),
            (np.int64(43), 43),
            (torch.tensor(6), 6),
            (torch.tensor([46]), 46),
        ],
    )
    def test_integer_values_are_their_own_index(self, value, expected):
        result = AttributeToClassTransform("h11")(Data(h11=value))
        assert result.y.item() == expected
        assert result.y.dtype == torch.long
        assert result.y.shape == ()

    def test_index_is_independent_of_sample_order(self):
        # Statelessness: the same value yields the same index no matter
        # which samples were seen before (unlike encounter-order
        # enumeration, which would yield [0, 1, 0, 2, 1] here).
        transform = AttributeToClassTransform("h11")
        ys = [transform(Data(h11=v)).y.item() for v in (7, 3, 7, 11, 3)]
        assert ys == [7, 3, 7, 11, 3]

    def test_explicit_mapping(self):
        transform = AttributeToClassTransform("kind", mapping={"a": 0, "b": 1})
        assert transform(Data(kind="b")).y.item() == 1
        assert transform.num_classes == 2

    def test_mapping_with_tensor_value(self):
        transform = AttributeToClassTransform("h11", mapping={6: 0, 7: 1})
        assert transform(Data(h11=torch.tensor(7))).y.item() == 1

    def test_unknown_mapping_value_raises(self):
        transform = AttributeToClassTransform("kind", mapping={"a": 0})
        with pytest.raises(KeyError, match="Unknown value 'c'"):
            transform(Data(kind="c"))

    def test_non_integer_without_mapping_raises(self):
        transform = AttributeToClassTransform("h11")
        with pytest.raises(TypeError, match="integer-valued"):
            transform(Data(h11=1.5))
        with pytest.raises(TypeError, match="integer-valued"):
            transform(Data(h11="S^2"))

    def test_non_scalar_tensor_raises(self):
        transform = AttributeToClassTransform("betti_numbers")
        with pytest.raises(AssertionError, match="scalar"):
            transform(Data(betti_numbers=torch.tensor([1, 0, 1])))

    def test_missing_source_raises(self):
        with pytest.raises(AssertionError, match="not present"):
            AttributeToClassTransform("h11")(Data())

    def test_num_classes_without_mapping(self):
        assert AttributeToClassTransform("h11").num_classes is None


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
    def test_vector_target(self):
        transform = AttributeToRegressionTransform(["h11", "h12"])
        result = transform(Data(h11=6, h12=46))
        assert result.y.dtype == torch.float32
        assert result.y.shape == (1, 2)
        assert result.y.tolist() == [[6.0, 46.0]]

    def test_scalar_source(self):
        result = AttributeToRegressionTransform("h12")(
            Data(h12=torch.tensor(46))
        )
        assert result.y.shape == (1, 1)
        assert result.y.item() == 46.0

    def test_sum_sources(self):
        transform = AttributeToRegressionTransform(
            ["h11", "h12"], sum_sources=True
        )
        result = transform(Data(h11=7, h12=43))
        assert result.y.shape == (1, 1)
        assert result.y.item() == 50.0

    def test_accepts_numpy_scalars(self):
        result = AttributeToRegressionTransform("c2")(Data(c2=np.float64(2.5)))
        assert result.y.item() == 2.5

    def test_missing_source_raises(self):
        transform = AttributeToRegressionTransform(["h11", "h99"])
        with pytest.raises(AssertionError, match="not present"):
            transform(Data(h11=6))
