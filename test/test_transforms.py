"""Tests for transforms module."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from mantra.transforms.moment_curve_embedding import (
    MomentCurveEmbedding,
    _calculate_moment_curve,
    _propagate_values,
    _sample_from_special_orthogonal_group,
)
from mantra.transforms.create_labels import CreateLabels
from mantra.transforms.select_features import SelectFeatures
from mantra.transforms.select_attributes import SelectAttributes
from mantra.transforms.attribute_transform import (
    NodeIndex,
    RandomNodeFeatures,
    NodeDegreeTransform,
    NodeRandomTransform,
)
from mantra.transforms.structural_transforms import (
    TriangulationToFaceTransform,
    SetNumNodesTransform,
)
from mantra.transforms.task_transforms import (
    OrientableToClassTransform,
    BettiToClassTransform,
    NameToClass2MTransform,
)

TETRAHEDRON_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]


def _make_data(**kwargs):
    defaults = dict(
        triangulation=TETRAHEDRON_TRI,
        dimension=2,
        n_vertices=4,
        betti_numbers=[1, 0, 1],
        name="S^2",
    )
    defaults.update(kwargs)
    return Data(**defaults)


# --- Moment Curve ---


class TestCalculateMomentCurve:
    def test_shape(self):
        X = _calculate_moment_curve(5, 2)
        assert X.shape == (5, 5)  # n=5, 2*d+1=5

    def test_shape_3d(self):
        X = _calculate_moment_curve(8, 3)
        assert X.shape == (8, 7)  # n=8, 2*3+1=7

    def test_first_row_zeros(self):
        """First vertex (t=0) should have all zeros."""
        X = _calculate_moment_curve(5, 2)
        np.testing.assert_allclose(X[0], 0.0, atol=1e-10)


class TestSampleSO:
    def test_orthogonal(self):
        Q = _sample_from_special_orthogonal_group(3, rng=np.random.default_rng(42))
        np.testing.assert_allclose(Q @ Q.T, np.eye(3), atol=1e-10)

    def test_positive_determinant(self):
        Q = _sample_from_special_orthogonal_group(3, rng=np.random.default_rng(42))
        assert np.linalg.det(Q) > 0

    def test_invalid_dimension_raises(self):
        with pytest.raises(AssertionError):
            _sample_from_special_orthogonal_group(0)


class TestMomentCurveEmbedding:
    def test_basic(self):
        data = _make_data()
        transform = MomentCurveEmbedding()
        result = transform(data)
        assert "moment_curve_embedding" in result
        assert result.moment_curve_embedding.shape == (4, 5)

    def test_perturb(self):
        data = _make_data()
        t1 = MomentCurveEmbedding(perturb=False)
        t2 = MomentCurveEmbedding(perturb=True, rng=42)
        r1 = t1(data)
        r2 = t2(_make_data())
        # Perturbed should differ
        assert not torch.allclose(
            r1.moment_curve_embedding, r2.moment_curve_embedding
        )

    def test_normalize_on_sphere(self):
        data = _make_data()
        transform = MomentCurveEmbedding(normalize=True)
        result = transform(data)
        emb = result.moment_curve_embedding
        # All points should have norm 1 (on a sphere)
        norms = torch.norm(emb, dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)

    def test_propagate(self):
        data = _make_data()
        transform = MomentCurveEmbedding(propagate=True)
        result = transform(data)
        emb = result.moment_curve_embedding
        assert isinstance(emb, dict)
        assert 0 in emb  # vertices
        assert 1 in emb  # edges

    def test_seed_reproducible(self):
        r1 = MomentCurveEmbedding(perturb=True, rng=42)(_make_data())
        r2 = MomentCurveEmbedding(perturb=True, rng=42)(_make_data())
        torch.testing.assert_close(
            r1.moment_curve_embedding, r2.moment_curve_embedding
        )


class TestPropagateValues:
    def test_barycenter(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        tri = [[1, 2, 3]]
        values = _propagate_values(X, tri)
        # edge barycenters
        assert values[1].shape[0] == 3  # 3 edges
        # triangle barycenter
        assert values[0].shape[0] == 3  # 3 vertices


# --- Create Labels ---


class TestCreateLabels:
    def test_string_labels(self):
        transform = CreateLabels(source="name")
        d1 = transform(_make_data(name="S^2"))
        d2 = transform(_make_data(name="T^2"))
        d3 = transform(_make_data(name="S^2"))
        assert d1.y.item() != d2.y.item()
        assert d1.y.item() == d3.y.item()

    def test_boolean_label_true(self):
        transform = CreateLabels(source="orientable")
        result = transform(_make_data(orientable=True))
        assert result.y.item() == 1

    def test_boolean_label_false(self):
        transform = CreateLabels(source="orientable")
        result = transform(_make_data(orientable=False))
        assert result.y.item() == 0

    def test_missing_source_raises(self):
        transform = CreateLabels(source="nonexistent")
        with pytest.raises(AssertionError):
            transform(_make_data())


# --- Select Features ---


class TestSelectFeatures:
    def test_graph_mode(self):
        data = _make_data()
        data.moment_curve_embedding = torch.randn(4, 5)
        transform = SelectFeatures(src="moment_curve_embedding", dst="x")
        result = transform(data)
        assert "x" in result
        assert result.x.shape == (4, 5)

    def test_sc_mode(self):
        data = _make_data()
        data.my_features = {0: torch.randn(4, 3), 1: torch.randn(6, 3)}
        transform = SelectFeatures(
            src="my_features", dst=None, representation="sc"
        )
        result = transform(data)
        assert "x_0" in result
        assert "x_1" in result


# --- Select Attributes ---


class TestSelectAttributes:
    def test_keeps_specified_keys(self):
        data = _make_data()
        data.x = torch.randn(4, 3)
        data.y = torch.tensor([0])
        data.edge_index = torch.tensor([[0, 1], [1, 0]])
        transform = SelectAttributes(keep_keys=["x", "y", "edge_index"])
        result = transform(data)
        assert "x" in result
        assert "y" in result

    def test_default_keys(self):
        transform = SelectAttributes()
        assert transform.keep_keys == {"x", "y", "edge_index"}


# --- Structural Transforms ---


class TestNodeIndex:
    def test_creates_face_tensor(self):
        data = _make_data()
        transform = NodeIndex()
        result = transform(data)
        assert hasattr(result, "face")
        # Faces should be 0-indexed
        assert result.face.min().item() == 0
        assert result.face.max().item() == 3  # was 4, minus 1


class TestRandomNodeFeatures:
    def test_creates_features(self):
        data = _make_data()
        data = NodeIndex()(data)
        transform = RandomNodeFeatures(dimension=8)
        result = transform(data)
        assert result.x.shape == (4, 8)


class TestNodeDegreeTransform:
    def test_creates_degree(self):
        data = _make_data()
        data = NodeIndex()(data)
        from torch_geometric.transforms import FaceToEdge

        data = FaceToEdge(remove_faces=False)(data)
        transform = NodeDegreeTransform()
        result = transform(data)
        assert hasattr(result, "degree")
        assert result.degree.shape[0] == 4


class TestNodeRandomTransform:
    def test_creates_random_features(self):
        data = _make_data()
        data = NodeIndex()(data)
        from torch_geometric.transforms import FaceToEdge

        data = FaceToEdge(remove_faces=False)(data)
        transform = NodeRandomTransform(dim=5)
        result = transform(data)
        assert result.random_features.shape[1] == 5


# --- Attribute Transforms ---


class TestTriangulationToFaceTransform:
    def test_creates_faces(self):
        data = _make_data()
        data = NodeIndex()(data)
        transform = TriangulationToFaceTransform()
        result = transform(data)
        assert hasattr(result, "face")

    def test_remove_triangulation(self):
        data = _make_data()
        data = NodeIndex()(data)
        transform = TriangulationToFaceTransform(remove_triangulation=True)
        result = transform(data)
        # PyG may delete the attr or set it to None depending on version
        tri = getattr(result, "triangulation", None)
        assert tri is None


class TestSetNumNodesTransform:
    def test_sets_num_nodes(self):
        data = _make_data()
        transform = SetNumNodesTransform()
        result = transform(data)
        assert result.num_nodes == 4


# --- Task Transforms ---


class TestOrientableToClassTransform:
    def test_orientable(self):
        data = _make_data(betti_numbers=[1, 0, 1])
        transform = OrientableToClassTransform()
        result = transform(data)
        assert result.y.item() == 1  # last betti = 1 -> orientable

    def test_non_orientable(self):
        data = _make_data(betti_numbers=[1, 0, 0])
        transform = OrientableToClassTransform()
        result = transform(data)
        assert result.y.item() == 0


class TestBettiToClassTransform:
    def test_2d(self):
        data = _make_data(betti_numbers=[1, 2, 1])
        transform = BettiToClassTransform(manifold_dim=2)
        result = transform(data)
        assert result.y.shape == (1, 3)

    def test_3d(self):
        data = _make_data(betti_numbers=[1, 0, 0, 1])
        transform = BettiToClassTransform(manifold_dim=3)
        result = transform(data)
        assert result.y.shape == (1, 4)

    def test_invalid_dim_raises(self):
        with pytest.raises(AssertionError):
            BettiToClassTransform(manifold_dim=4)


class TestNameToClass2MTransform:
    def test_known_names(self):
        transform = NameToClass2MTransform()
        for name, expected_class in {
            "S^2": 3,
            "T^2": 2,
            "RP^2": 1,
        }.items():
            data = _make_data(name=name)
            result = transform.forward(data)
            assert result.y.item() == expected_class
