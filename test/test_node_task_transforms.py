"""Tests for ``mantra.transforms.node_task_transforms``."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from mantra.transforms import (
    AttributeToNodeClassTransform,
    AttributeToNodeRegressionTransform,
)


def _graph(n, offset, **attributes):
    """Cycle graph on `n` nodes whose feature is the local node index.

    Together with the graph-specific `offset` this lets a test detect
    any misalignment between `y`, `x`, `batch` and `edge_index`.
    """
    return Data(
        x=torch.arange(n, dtype=torch.float).view(-1, 1),
        edge_index=torch.tensor([[i, (i + 1) % n] for i in range(n)]).t(),
        offset=offset,
        **attributes,
    )


class TestAttributeToNodeRegressionTransform:
    @pytest.mark.parametrize(
        "value",
        [
            # `pre_transform` path: a plain list of scalars.
            [4, -2, 6],
            # `transform` path: tensors, of any numeric dtype.
            torch.tensor([4, -2, 6]),
            torch.tensor([4.0, -2.0, 6.0], dtype=torch.float64),
        ],
    )
    def test_one_value_per_vertex(self, value):
        result = AttributeToNodeRegressionTransform("value")(Data(value=value))

        assert result.y.dtype == torch.float32
        assert result.y.shape == (3, 1)
        assert result.y.flatten().tolist() == [4.0, -2.0, 6.0]

    def test_source_is_left_in_place(self):
        value = torch.tensor([1, 2])

        data = AttributeToNodeRegressionTransform("value")(Data(value=value))

        assert data.value is value

    def test_batches_by_concatenating_vertices(self):
        transform = AttributeToNodeRegressionTransform("value")

        batch = Batch.from_data_list(
            [
                transform(Data(value=torch.tensor([0, 1, 2]))),
                transform(Data(value=torch.tensor([0, 5]))),
            ]
        )

        assert batch.y.shape == (5, 1)
        assert batch.y.flatten().tolist() == [0.0, 1.0, 2.0, 0.0, 5.0]

    @pytest.mark.parametrize(
        "value",
        [torch.tensor([[1, 2, 3]]), torch.tensor(5), [[1, 2], [3, 4]]],
    )
    def test_non_vector_raises(self, value):
        transform = AttributeToNodeRegressionTransform("value")

        with pytest.raises(AssertionError, match="one value per vertex"):
            transform(Data(value=value))

    def test_missing_source_raises(self):
        with pytest.raises(KeyError):
            AttributeToNodeRegressionTransform("value")(Data(n_vertices=3))

    def test_targets_follow_node_order_through_batching(self):
        # Value = 10 * graph + local node index, so that every row of
        # `y` can be checked against the node it is supposed to belong
        # to, before and after batching.
        transform = AttributeToNodeRegressionTransform("value")
        graphs = [
            transform(_graph(3, offset=10, value=[10, 11, 12])),
            transform(_graph(2, offset=20, value=[20, 21])),
        ]

        for data in graphs:
            # Row i of `y` is the target of node i (the node with `x == i`).
            assert torch.equal(
                data.y.flatten(), data.offset + data.x.flatten()
            )

        batch = Batch.from_data_list(graphs)

        assert batch.ptr.tolist() == [0, 3, 5]
        assert batch.batch.tolist() == [0, 0, 0, 1, 1]
        # Row k of the batched `y` belongs to node k: graph `batch[k]`,
        # local index `x[k]`.
        expected = 10 * (batch.batch + 1) + batch.x.flatten().long()
        assert batch.y.flatten().tolist() == expected.tolist()
        # Edge endpoints, offset by the batching, index the right targets.
        src, dst = batch.edge_index
        assert torch.equal(batch.y[src].flatten(), expected[src].float())
        assert torch.equal(batch.y[dst].flatten(), expected[dst].float())


class TestAttributeToNodeClassTransform:
    MAPPING = {3: 0, 7: 1, 11: 2}

    def test_values_are_mapped_to_class_indices(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)

        result = transform(Data(label=torch.tensor([7, 3, 7, 11])))

        assert result.y.dtype == torch.long
        assert result.y.shape == (4,)
        assert result.y.tolist() == [1, 0, 1, 2]

    def test_accepts_list_of_values(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)

        assert transform(Data(label=[11, 3])).y.tolist() == [2, 0]

    def test_index_is_independent_of_sample_order(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)

        first = transform(Data(label=torch.tensor([11, 3]))).y.tolist()
        second = transform(Data(label=torch.tensor([3, 11]))).y.tolist()

        assert first == [2, 0]
        assert second == [0, 2]

    def test_num_classes(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)

        assert transform.num_classes == 3

    def test_batches_by_concatenating_vertices(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)

        batch = Batch.from_data_list(
            [
                transform(Data(label=torch.tensor([3, 7]))),
                transform(Data(label=torch.tensor([11]))),
            ]
        )

        assert batch.y.tolist() == [0, 1, 2]

    def test_unknown_value_raises(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)

        with pytest.raises(KeyError, match="Unknown value 5"):
            transform(Data(label=torch.tensor([3, 5])))

    def test_float_values_raise(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)

        with pytest.raises(AssertionError, match="type int"):
            transform(Data(label=torch.tensor([3.0, 7.0])))

    def test_missing_source_raises(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)

        with pytest.raises(KeyError):
            transform(Data())

    def test_indices_follow_node_order_through_batching(self):
        transform = AttributeToNodeClassTransform("label", self.MAPPING)
        batch = Batch.from_data_list(
            [
                transform(_graph(3, offset=0, label=[3, 7, 11])),
                transform(_graph(2, offset=0, label=[11, 3])),
            ]
        )

        assert batch.batch.tolist() == [0, 0, 0, 1, 1]
        # Concatenated in graph order, one index per node.
        assert batch.y.tolist() == [0, 1, 2, 2, 0]
        # Node k with local index `x[k]` carries the label of that node.
        local = batch.x.flatten().long()
        labels = torch.tensor([3, 7, 11, 11, 3])
        assert torch.equal(
            batch.y, torch.tensor([self.MAPPING[int(v)] for v in labels])
        )
        assert torch.equal(local, torch.tensor([0, 1, 2, 0, 1]))
