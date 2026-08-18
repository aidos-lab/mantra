"""Tests for ``mantra.transforms.task_transforms``."""

import pytest
from torch_geometric.data import Data

from mantra.manifold_types import Manifold2Type
from mantra.transforms.task_transforms import (
    NAME_TO_CLASS_2M,
    BettiToClassTransform,
    NameToClass2MTransform,
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
        # 21 enum classes -> indices 0..20 with no gaps.
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
