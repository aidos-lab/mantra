"""Tests for ``mantra.transforms.task_transforms``."""

import pytest
import torch
from torch_geometric.data import Data

from mantra.manifold_types import Manifold2Type
from mantra.transforms.task_transforms import (
    NAME_TO_CLASS_2M,
    BettiToClassTransform,
    BinaryHomeomorphicTransform,
    NameToClass2MTransform,
    OrientableToClassTransform,
)


class TestNameToClass2M:
    def test_is_exactly_the_enum_values(self):
        # The map covers every enum value once and nothing else: no dead
        # "" entry and no "#^2 RP^2" alias (the dataset stores the Klein
        # bottle canonically).
        assert set(NAME_TO_CLASS_2M) == {m.value for m in Manifold2Type}
        assert "" not in NAME_TO_CLASS_2M
        assert "#^2 RP^2" not in NAME_TO_CLASS_2M

    def test_indices_are_contiguous_from_zero(self):
        # One index per enum class, contiguous from zero with no gaps.
        assert set(NAME_TO_CLASS_2M.values()) == set(range(len(Manifold2Type)))

    @pytest.mark.parametrize("name", [m.value for m in Manifold2Type])
    def test_transform_matches_enum_order(self, name):
        # The class assigned to a name is its position in ``Manifold2Type``.
        expected = [m.value for m in Manifold2Type].index(name)
        result = NameToClass2MTransform().forward(Data(name=name))
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

    def test_callable_like_a_basetransform(self):
        # Regression: the transform must be usable as a callable (e.g. inside
        # a ``Compose`` / dataset pipeline), not only via ``.forward``. It
        # subclasses ``BaseTransform`` which supplies ``__call__``.
        result = NameToClass2MTransform()(Data(name="S^2"))
        assert result.y.item() == 0


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


class TestBinaryHomeomorphicTransform:
    def _pair(self, name_a, name_b):
        # A pairwise object: first axis must have length 2.
        return Data(
            triangulation=torch.zeros(2, 3, dtype=torch.long),
            name=[name_a, name_b],
        )

    def test_same_name_is_homeomorphic(self):
        result = BinaryHomeomorphicTransform()(self._pair("S^2", "S^2"))
        assert result.y.item() == 1

    def test_different_name_is_not_homeomorphic(self):
        result = BinaryHomeomorphicTransform()(self._pair("S^2", "T^2"))
        assert result.y.item() == 0

    def test_requires_triangulation(self):
        with pytest.raises(AssertionError, match="No triangulation"):
            BinaryHomeomorphicTransform()(Data(name=["S^2", "S^2"]))

    def test_requires_pairwise_first_axis(self):
        data = Data(
            triangulation=torch.zeros(1, 3, dtype=torch.long),
            name=["S^2"],
        )
        with pytest.raises(AssertionError, match="pairwise"):
            BinaryHomeomorphicTransform()(data)
