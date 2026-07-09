"""Tests for ``mantra.transforms.moment_curve_embedding``."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from mantra.transforms import MomentCurveEmbedding
from mantra.transforms.moment_curve_embedding import (
    _calculate_moment_curve,
    _propagate_values,
    _sample_from_special_orthogonal_group,
)

# Boundary of a tetrahedron: a triangulated 2-sphere with
# 4 vertices, 6 edges and 4 triangular faces.
TETRAHEDRON_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]


def _make_data(triangulation, n_vertices, dimension):
    return Data(
        triangulation=triangulation,
        n_vertices=n_vertices,
        dimension=dimension,
    )


class TestCalculateMomentCurve:
    def test_shape(self):
        X = _calculate_moment_curve(n=5, d=2)
        # Coordinates live in dimension 2 * d + 1.
        assert X.shape == (5, 5)

    def test_endpoints(self):
        X = _calculate_moment_curve(n=4, d=1)
        # t = 0 maps to the origin...
        np.testing.assert_allclose(X[0], np.zeros(3))
        # ...and t = 1 maps to the all-ones vector.
        np.testing.assert_allclose(X[-1], np.ones(3))

    def test_points_are_distinct(self):
        X = _calculate_moment_curve(n=10, d=3)
        # The moment curve does not self-intersect: all points must
        # be pairwise distinct.
        assert len(np.unique(X, axis=0)) == X.shape[0]


class TestPropagateValues:
    def test_keys_present_for_every_dimension(self):
        X = np.arange(4 * 3, dtype=float).reshape(4, 3)
        values = _propagate_values(X, TETRAHEDRON_TRI)

        assert set(values.keys()) == {0, 1, 2}

    def test_vertex_values_unchanged(self):
        X = np.arange(4 * 3, dtype=float).reshape(4, 3)
        values = _propagate_values(X, TETRAHEDRON_TRI)

        np.testing.assert_allclose(values[0].numpy(), X)

    def test_shapes(self):
        X = np.arange(4 * 3, dtype=float).reshape(4, 3)
        values = _propagate_values(X, TETRAHEDRON_TRI)

        # 4 vertices, 6 edges, 4 triangles.
        assert values[0].shape == (4, 3)
        assert values[1].shape == (6, 3)
        assert values[2].shape == (4, 3)

    def test_edge_barycenter(self):
        X = np.asarray(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
        )
        values = _propagate_values(X, TETRAHEDRON_TRI)

        # Edge (1, 2) is the first edge in lexicographic order; its
        # barycenter is the midpoint of vertices 1 and 2 (1-indexed).
        np.testing.assert_allclose(values[1][0].numpy(), [1.0, 0.0])

    def test_face_barycenter(self):
        X = np.asarray(
            [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]],
        )
        values = _propagate_values(X, TETRAHEDRON_TRI)

        # Face (1, 2, 3) is the first triangle; its barycenter is the
        # mean of vertices 1, 2 and 3.
        np.testing.assert_allclose(values[2][0].numpy(), [1.0, 1.0])


class TestSampleFromSpecialOrthogonalGroup:
    def test_shape(self):
        rng = np.random.default_rng(0)
        Q = _sample_from_special_orthogonal_group(5, rng=rng)
        assert Q.shape == (5, 5)

    def test_orthogonal(self):
        rng = np.random.default_rng(0)
        Q = _sample_from_special_orthogonal_group(4, rng=rng)
        np.testing.assert_allclose(Q @ Q.T, np.eye(4), atol=1e-10)

    def test_determinant_is_one(self):
        rng = np.random.default_rng(0)
        Q = _sample_from_special_orthogonal_group(6, rng=rng)
        assert np.linalg.det(Q) == pytest.approx(1.0)

    def test_rejects_non_positive_dimension(self):
        with pytest.raises(AssertionError):
            _sample_from_special_orthogonal_group(0)


class TestMomentCurveEmbedding:
    def test_requires_n_vertices_and_dimension(self):
        data = Data(triangulation=TETRAHEDRON_TRI)
        with pytest.raises(AssertionError):
            MomentCurveEmbedding()(data)

    def test_default_shape(self):
        data = _make_data(TETRAHEDRON_TRI, n_vertices=4, dimension=2)
        result = MomentCurveEmbedding(rng=0)(data)

        # 2 * d + 1 = 5 coordinates, one row per vertex.
        assert result.moment_curve_embedding.shape == (4, 5)
        assert result.moment_curve_embedding.dtype == torch.float32

    def test_does_not_overwrite_other_attributes(self):
        data = _make_data(TETRAHEDRON_TRI, n_vertices=4, dimension=2)
        result = MomentCurveEmbedding(rng=0)(data)

        assert result.triangulation == TETRAHEDRON_TRI
        assert result.n_vertices == 4

    def test_normalize_adds_one_dimension(self):
        data = _make_data(TETRAHEDRON_TRI, n_vertices=4, dimension=2)
        result = MomentCurveEmbedding(normalize=True, rng=0)(data)

        assert result.moment_curve_embedding.shape == (4, 6)

    def test_normalize_places_points_on_unit_sphere(self):
        data = _make_data(TETRAHEDRON_TRI, n_vertices=4, dimension=2)
        result = MomentCurveEmbedding(normalize=True, rng=0)(data)

        norms = torch.linalg.norm(result.moment_curve_embedding, dim=1)
        np.testing.assert_allclose(norms.numpy(), np.ones(4), atol=1e-6)

    def test_perturb_preserves_pairwise_distances(self):
        data_a = _make_data(TETRAHEDRON_TRI, n_vertices=5, dimension=2)
        data_b = _make_data(TETRAHEDRON_TRI, n_vertices=5, dimension=2)

        unperturbed = MomentCurveEmbedding()(data_a).moment_curve_embedding
        perturbed = MomentCurveEmbedding(perturb=True, rng=0)(
            data_b
        ).moment_curve_embedding

        def pairwise_distances(X):
            return torch.cdist(X, X)

        np.testing.assert_allclose(
            pairwise_distances(unperturbed).numpy(),
            pairwise_distances(perturbed).numpy(),
            atol=1e-5,
        )

    def test_perturb_changes_coordinates(self):
        data = _make_data(TETRAHEDRON_TRI, n_vertices=5, dimension=2)
        unperturbed = _calculate_moment_curve(5, 2)

        result = MomentCurveEmbedding(perturb=True, rng=0)(data)

        assert not np.allclose(
            result.moment_curve_embedding.numpy(), unperturbed
        )

    def test_perturb_is_reproducible_with_seed(self):
        data_a = _make_data(TETRAHEDRON_TRI, n_vertices=5, dimension=2)
        data_b = _make_data(TETRAHEDRON_TRI, n_vertices=5, dimension=2)

        result_a = MomentCurveEmbedding(perturb=True, rng=42)(data_a)
        result_b = MomentCurveEmbedding(perturb=True, rng=42)(data_b)

        np.testing.assert_allclose(
            result_a.moment_curve_embedding.numpy(),
            result_b.moment_curve_embedding.numpy(),
        )
