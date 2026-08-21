"""Tests for ``mantra.transforms.node_task_transforms``."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from mantra.transforms import (
    AttributeToNodeClassTransform,
    AttributeToNodeRegressionTransform,
)


class TestAttributeToNodeRegressionTransform:
    @pytest.mark.parametrize(
        "value",
        [
            torch.tensor([4, -2, 6]),
            torch.tensor([4.0, -2.0, 6.0], dtype=torch.float64),
        ],
    )
    def test_one_value_per_vertex(self, value):
        result = AttributeToNodeRegressionTransform("c2")(
            Data(c2=value, n_vertices=3)
        )
        assert result.y.dtype == torch.float32
        assert result.y.shape == (3, 1)
        assert result.y.flatten().tolist() == [4.0, -2.0, 6.0]
        assert result.node_mask.dtype == torch.bool
        assert result.node_mask.tolist() == [True, True, True]

    def test_mask_first_excludes_first_vertex(self):
        result = AttributeToNodeRegressionTransform("c2", mask_first=True)(
            Data(c2=torch.tensor([0, 1, 2, 3]), n_vertices=4)
        )
        assert result.node_mask.tolist() == [False, True, True, True]
        # The target keeps every vertex; masking is left to the consumer.
        assert result.y.shape == (4, 1)
        assert result.y[result.node_mask].flatten().tolist() == [1.0, 2.0, 3.0]

    def test_source_is_left_in_place(self):
        value = torch.tensor([1, 2])
        data = AttributeToNodeRegressionTransform("c2")(Data(c2=value))
        assert data.c2 is value

    def test_without_n_vertices(self):
        # The consistency check only runs if `n_vertices` is available.
        result = AttributeToNodeRegressionTransform("c2")(
            Data(c2=torch.tensor([1, 2]))
        )
        assert result.y.shape == (2, 1)

    def test_accepts_tensor_n_vertices(self):
        result = AttributeToNodeRegressionTransform("c2")(
            Data(c2=torch.tensor([1, 2]), n_vertices=torch.tensor(2))
        )
        assert result.y.shape == (2, 1)

    def test_batches_by_concatenating_vertices(self):
        transform = AttributeToNodeRegressionTransform("c2", mask_first=True)
        batch = Batch.from_data_list(
            [
                transform(Data(c2=torch.tensor([0, 1, 2]), n_vertices=3)),
                transform(Data(c2=torch.tensor([0, 5]), n_vertices=2)),
            ]
        )
        assert batch.y.shape == (5, 1)
        assert batch.node_mask.tolist() == [False, True, True, False, True]
        assert batch.y[batch.node_mask].flatten().tolist() == [1.0, 2.0, 5.0]

    def test_vertex_count_mismatch_raises(self):
        transform = AttributeToNodeRegressionTransform("c2")
        with pytest.raises(AssertionError, match="holds 2 values for 3"):
            transform(Data(c2=torch.tensor([1, 2]), n_vertices=3))

    @pytest.mark.parametrize("value", [[1, 2, 3], np.array([1, 2, 3])])
    def test_non_tensor_raises(self, value):
        transform = AttributeToNodeRegressionTransform("c2")
        with pytest.raises(AssertionError, match="must be a tensor"):
            transform(Data(c2=value, n_vertices=3))

    @pytest.mark.parametrize(
        "value", [torch.tensor([[1, 2, 3]]), torch.tensor(5)]
    )
    def test_non_vector_raises(self, value):
        transform = AttributeToNodeRegressionTransform("c2")
        with pytest.raises(AssertionError, match="one value per vertex"):
            transform(Data(c2=value))

    def test_missing_source_raises(self):
        with pytest.raises(AssertionError, match="not present"):
            AttributeToNodeRegressionTransform("c2")(Data(n_vertices=3))


class TestAttributeToNodeClassTransform:
    MAPPING = {3: 0, 7: 1, 11: 2}

    def test_values_are_mapped_to_class_indices(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)
        result = transform(
            Data(label=torch.tensor([7, 3, 7, 11]), n_vertices=4)
        )
        assert result.y.dtype == torch.long
        assert result.y.shape == (4,)
        assert result.y.tolist() == [1, 0, 1, 2]
        assert result.node_mask.tolist() == [True] * 4

    def test_index_is_independent_of_sample_order(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)
        first = transform(Data(label=torch.tensor([11, 3]))).y.tolist()
        second = transform(Data(label=torch.tensor([3, 11]))).y.tolist()
        assert first == [2, 0]
        assert second == [0, 2]

    def test_num_classes(self):
        assert (
            AttributeToNodeClassTransform("label", self.MAPPING).num_classes
            == 3
        )

    def test_mask_first_excludes_first_vertex(self):
        transform = AttributeToNodeClassTransform(
            "label", self.MAPPING, mask_first=True
        )
        result = transform(Data(label=torch.tensor([3, 7, 11])))
        assert result.node_mask.tolist() == [False, True, True]
        assert result.y[result.node_mask].tolist() == [1, 2]

    def test_batches_by_concatenating_vertices(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)
        batch = Batch.from_data_list(
            [
                transform(Data(label=torch.tensor([3, 7]))),
                transform(Data(label=torch.tensor([11]))),
            ]
        )
        assert batch.y.tolist() == [0, 1, 2]
        assert batch.node_mask.shape == (3,)

    def test_unknown_value_raises(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)
        with pytest.raises(KeyError) as excinfo:
            transform(Data(label=torch.tensor([3, 5])))
        assert excinfo.value.args[0] == 5

    def test_non_tensor_raises(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)
        with pytest.raises(AssertionError, match="must be a tensor"):
            transform(Data(label=[3, 7]))

    def test_missing_source_raises(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)
        with pytest.raises(AssertionError, match="not present"):
            transform(Data())
