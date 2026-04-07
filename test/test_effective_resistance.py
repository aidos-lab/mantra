"""Tests for effective resistance transforms."""

import numpy as np
import torch
import pytest
from torch_geometric.data import Data

from mantra.transforms.effective_resistance import (
    weighted_chain_laplacian,
    calculate_er,
    er_statistics,
    EffectiveResistanceEmbedding,
    EffectiveResistanceStatisticsEmbedding,
)
from mantra.representations.simplicial_connectivity import (
    IncidenceSimplicialComplex,
)

# Boundary of tetrahedron
TETRAHEDRON_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]


def _make_data_with_incidence(triangulation=None, dimension=2):
    if triangulation is None:
        triangulation = TETRAHEDRON_TRI
    n_vertices = len(set(v for s in triangulation for v in s))
    data = Data(
        triangulation=triangulation,
        dimension=dimension,
        n_vertices=n_vertices,
    )
    transform = IncidenceSimplicialComplex(signed=True)
    return transform(data)


class TestWeightedChainLaplacian:
    def test_returns_three_matrices(self):
        # Simple 3x3 identity-like setup
        B_p = torch.zeros((1, 3))
        B_p_plus_1 = torch.eye(3)
        W_p = torch.eye(3)
        W_p_plus_1 = torch.eye(3)
        W_p_minus_1 = torch.ones((1, 1))

        L_up, L_down, L_hodge = weighted_chain_laplacian(
            B_p, B_p_plus_1, W_p, W_p_plus_1, W_p_minus_1
        )
        assert L_up.shape == (3, 3)
        assert L_down.shape == (3, 3)
        assert L_hodge.shape == (3, 3)

    def test_hodge_is_sum(self):
        B_p = torch.zeros((1, 3))
        B_p_plus_1 = torch.eye(3)
        W_p = torch.eye(3)
        W_p_plus_1 = torch.eye(3)
        W_p_minus_1 = torch.ones((1, 1))

        L_up, L_down, L_hodge = weighted_chain_laplacian(
            B_p, B_p_plus_1, W_p, W_p_plus_1, W_p_minus_1
        )
        np.testing.assert_allclose(
            L_hodge.numpy(), (L_up + L_down).numpy(), atol=1e-6
        )


class TestCalculateER:
    def test_output_shape(self):
        data = _make_data_with_incidence()
        B1 = data.incidence_1.to_dense()
        W_0 = torch.eye(B1.shape[0])
        B_0 = torch.zeros((1, B1.shape[0]))
        W_minus_1 = torch.ones((1, 1))

        L_up_0, _, _ = weighted_chain_laplacian(
            B_0,
            B1,
            W_0,
            torch.eye(B1.shape[1]),
            W_minus_1,
        )
        R = calculate_er(B1, W_0, L_up_0)
        # Should have one value per edge
        assert R.shape == (6,)

    def test_values_positive(self):
        data = _make_data_with_incidence()
        B1 = data.incidence_1.to_dense()
        W_0 = torch.eye(B1.shape[0])
        B_0 = torch.zeros((1, B1.shape[0]))
        W_minus_1 = torch.ones((1, 1))

        L_up_0, _, _ = weighted_chain_laplacian(
            B_0,
            B1,
            W_0,
            torch.eye(B1.shape[1]),
            W_minus_1,
        )
        R = calculate_er(B1, W_0, L_up_0)
        assert (R >= -1e-6).all()  # allow tiny numerical noise


class TestERStatistics:
    def test_default_statistics(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = er_statistics(x)
        assert stats.shape == (7,)

    def test_mean_is_correct(self):
        x = torch.tensor([2.0, 4.0, 6.0])
        stats = er_statistics(x, statistics_to_compute=["mean"])
        torch.testing.assert_close(stats, torch.tensor([4.0]))

    def test_min_max(self):
        x = torch.tensor([1.0, 5.0, 3.0])
        stats = er_statistics(x, statistics_to_compute=["min", "max"])
        assert stats[0].item() == 1.0
        assert stats[1].item() == 5.0

    def test_constant_std_is_zero(self):
        x = torch.tensor([3.0, 3.0, 3.0])
        stats = er_statistics(x, statistics_to_compute=["std"])
        assert stats[0].item() == 0.0


class TestEffectiveResistanceEmbedding:
    def test_creates_er_attribute(self):
        data = _make_data_with_incidence()
        transform = EffectiveResistanceEmbedding()
        result = transform(data)
        assert hasattr(result, "er")
        assert isinstance(result.er, dict)
        assert 1 in result.er  # edge effective resistances


class TestEffectiveResistanceStatisticsEmbedding:
    def test_creates_er_stats(self):
        data = _make_data_with_incidence()
        transform = EffectiveResistanceStatisticsEmbedding(
            dimensions_to_compute=[1]
        )
        result = transform(data)
        assert hasattr(result, "er_stats")
        # 1 dimension * 7 statistics = 7, flattened + unsqueezed
        assert result.er_stats.shape == (1, 7)
