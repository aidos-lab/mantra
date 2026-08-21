"""Tests for ``mantra.transforms.encodings.ScalarFeatures``."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from mantra.transforms import ScalarFeatures


class TestScalarFeatures:
    def test_collects_python_scalars(self):
        # Attributes as they arrive in `pre_transform` (raw `process()`).
        out = ScalarFeatures(["genus", "n_vertices"])(
            Data(genus=6, n_vertices=5, dimension=2)
        )

        assert out.scalar_features.dtype == torch.float32
        assert out.scalar_features.shape == (1, 2)
        assert out.scalar_features.tolist() == [[6.0, 5.0]]

    def test_collects_one_element_tensors(self):
        # Attributes as they arrive in `transform` (after collation).
        out = ScalarFeatures(["n_vertices", "genus"])(
            Data(n_vertices=torch.tensor([6]), genus=torch.tensor(2))
        )

        assert out.scalar_features.tolist() == [[6.0, 2.0]]

    def test_single_source_as_string(self):
        out = ScalarFeatures("genus")(Data(genus=6))

        assert out.scalar_features.shape == (1, 1)
        assert out.scalar_features.tolist() == [[6.0]]

    def test_order_follows_sources(self):
        data = Data(a=1, b=2, c=3)

        assert ScalarFeatures(["c", "a", "b"])(
            data
        ).scalar_features.tolist() == [[3.0, 1.0, 2.0]]

    def test_sources_are_left_in_place(self):
        out = ScalarFeatures(["genus"])(Data(genus=6))

        assert int(out.genus) == 6

    def test_batches_to_one_row_per_sample(self):
        transform = ScalarFeatures(["genus", "n_vertices"])
        batch = Batch.from_data_list(
            [
                transform(Data(genus=6, n_vertices=46)),
                transform(Data(genus=7, n_vertices=43)),
            ]
        )

        assert batch.scalar_features.shape == (2, 2)
        assert batch.scalar_features.tolist() == [[6.0, 46.0], [7.0, 43.0]]

    def test_missing_source_raises(self):
        with pytest.raises(AssertionError, match="not present"):
            ScalarFeatures(["genus", "missing"])(Data(genus=6))

    def test_non_scalar_tensor_raises(self):
        with pytest.raises(RuntimeError):
            ScalarFeatures(["coords"])(Data(coords=torch.zeros(3, 2)))
