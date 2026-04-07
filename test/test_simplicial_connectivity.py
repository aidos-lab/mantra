"""Tests for simplicial connectivity matrices."""

import pytest
import torch
from torch_geometric.data import Data

from mantra.representations.simplicial_connectivity import (
    AddSimplexTrie,
    IncidenceSimplicialComplex,
    UpLaplacianSimplicialComplex,
    DownLaplacianSimplicialComplex,
    HodgeLaplacianSimplicialComplex,
    AdjacencySimplicialComplex,
    CoadjacencySimplicialComplex,
)

# Boundary of tetrahedron: 4 vertices, 6 edges, 4 faces
TETRAHEDRON_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]

# Two triangles sharing an edge
TWO_TRIANGLES = [[1, 2, 3], [1, 2, 4]]


def _make_data(triangulation, dimension=2):
    n_vertices = len(set(v for s in triangulation for v in s))
    return Data(
        triangulation=triangulation,
        dimension=dimension,
        n_vertices=n_vertices,
    )


class TestAddSimplexTrie:
    def test_adds_trie(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = AddSimplexTrie()
        result = transform(data)
        assert hasattr(result, "simplex_trie")

    def test_trie_has_correct_shape(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = AddSimplexTrie()
        result = transform(data)
        # 4 vertices, 6 edges, 4 triangles
        assert result.simplex_trie.shape == [4, 6, 4]


class TestIncidenceMatrix:
    def test_incidence_1_shape(self):
        """B_1 should be (n_vertices, n_edges)."""
        data = _make_data(TETRAHEDRON_TRI)
        transform = IncidenceSimplicialComplex(signed=False)
        result = transform(data)
        inc = result.incidence_1.to_dense()
        assert inc.shape == (4, 6)

    def test_incidence_2_shape(self):
        """B_2 should be (n_edges, n_faces)."""
        data = _make_data(TETRAHEDRON_TRI)
        transform = IncidenceSimplicialComplex(signed=False)
        result = transform(data)
        inc = result.incidence_2.to_dense()
        assert inc.shape == (6, 4)

    def test_unsigned_incidence_values(self):
        """Unsigned incidence matrix should have only 0s and 1s."""
        data = _make_data(TETRAHEDRON_TRI)
        transform = IncidenceSimplicialComplex(signed=False)
        result = transform(data)
        inc = result.incidence_1.to_dense()
        assert set(inc.unique().tolist()).issubset({0.0, 1.0})

    def test_signed_incidence_values(self):
        """Signed incidence matrix should have -1, 0, 1."""
        data = _make_data(TETRAHEDRON_TRI)
        transform = IncidenceSimplicialComplex(signed=True)
        result = transform(data)
        inc = result.incidence_1.to_dense()
        assert set(inc.unique().tolist()).issubset({-1.0, 0.0, 1.0})

    def test_each_edge_has_two_vertices(self):
        """Each column of B_1 should have exactly 2 nonzero entries."""
        data = _make_data(TETRAHEDRON_TRI)
        transform = IncidenceSimplicialComplex(signed=False)
        result = transform(data)
        inc = result.incidence_1.to_dense()
        assert (inc.sum(dim=0) == 2).all()

    def test_each_face_has_three_edges(self):
        """Each column of B_2 should have exactly 3 nonzero entries."""
        data = _make_data(TETRAHEDRON_TRI)
        transform = IncidenceSimplicialComplex(signed=False)
        result = transform(data)
        inc = result.incidence_2.to_dense()
        assert (inc.sum(dim=0) == 3).all()

    def test_boundary_of_boundary_is_zero(self):
        """B_1 * B_2 should be the zero matrix (for signed)."""
        data = _make_data(TETRAHEDRON_TRI)
        transform = IncidenceSimplicialComplex(signed=True)
        result = transform(data)
        b1 = result.incidence_1.to_dense()
        b2 = result.incidence_2.to_dense()
        product = b1 @ b2
        assert torch.allclose(product, torch.zeros_like(product))

    def test_rank_0_raises(self):
        """Incidence at rank 0 should raise ValueError."""
        data = _make_data(TETRAHEDRON_TRI)
        data = AddSimplexTrie()(data)
        transform = IncidenceSimplicialComplex(signed=False)
        with pytest.raises(ValueError, match="Rank"):
            transform.generate_matrix(data.simplex_trie, 0, 2)


class TestUpLaplacian:
    def test_up_laplacian_shape(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = UpLaplacianSimplicialComplex(signed=True)
        result = transform(data)
        lap = result.up_laplacian_0.to_dense()
        # L_up_0 is (n_vertices x n_vertices)
        assert lap.shape == (4, 4)

    def test_up_laplacian_is_symmetric(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = UpLaplacianSimplicialComplex(signed=True)
        result = transform(data)
        lap = result.up_laplacian_0.to_dense()
        assert torch.allclose(lap, lap.T)

    def test_up_laplacian_is_graph_laplacian_for_rank_0(self):
        """L_up_0 = B_1 @ B_1^T should be the graph Laplacian."""
        data = _make_data(TETRAHEDRON_TRI)
        transform = UpLaplacianSimplicialComplex(signed=True)
        result = transform(data)
        lap = result.up_laplacian_0.to_dense()
        # For K4: diagonal should be 3 (degree), off-diagonal -1
        assert torch.allclose(torch.diag(lap), torch.tensor([3.0] * 4))


class TestDownLaplacian:
    def test_down_laplacian_shape(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = DownLaplacianSimplicialComplex(signed=True)
        result = transform(data)
        lap = result.down_laplacian_1.to_dense()
        # L_down_1 is (n_edges x n_edges)
        assert lap.shape == (6, 6)

    def test_down_laplacian_is_symmetric(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = DownLaplacianSimplicialComplex(signed=True)
        result = transform(data)
        lap = result.down_laplacian_1.to_dense()
        assert torch.allclose(lap, lap.T)


class TestHodgeLaplacian:
    def test_hodge_laplacian_rank0_equals_up_laplacian(self):
        """At rank 0, Hodge = Up Laplacian (no down component)."""
        data = _make_data(TETRAHEDRON_TRI)
        hodge_t = HodgeLaplacianSimplicialComplex(signed=True)
        up_t = UpLaplacianSimplicialComplex(signed=True)
        hodge_data = hodge_t(_make_data(TETRAHEDRON_TRI))
        up_data = up_t(_make_data(TETRAHEDRON_TRI))
        assert torch.allclose(
            hodge_data.hodge_laplacian_0.to_dense(),
            up_data.up_laplacian_0.to_dense(),
        )

    def test_hodge_laplacian_shape(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = HodgeLaplacianSimplicialComplex(signed=True)
        result = transform(data)
        # rank 1: (n_edges x n_edges)
        lap = result.hodge_laplacian_1.to_dense()
        assert lap.shape == (6, 6)

    def test_hodge_laplacian_is_symmetric(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = HodgeLaplacianSimplicialComplex(signed=True)
        result = transform(data)
        lap = result.hodge_laplacian_1.to_dense()
        assert torch.allclose(lap, lap.T)

    def test_hodge_laplacian_max_rank(self):
        """At max rank, Hodge = Down Laplacian (no up component)."""
        data = _make_data(TETRAHEDRON_TRI)
        hodge_t = HodgeLaplacianSimplicialComplex(signed=True)
        down_t = DownLaplacianSimplicialComplex(signed=True)
        hodge_data = hodge_t(_make_data(TETRAHEDRON_TRI))
        down_data = down_t(_make_data(TETRAHEDRON_TRI))
        assert torch.allclose(
            hodge_data.hodge_laplacian_2.to_dense(),
            down_data.down_laplacian_2.to_dense(),
        )


class TestAdjacency:
    def test_adjacency_shape(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = AdjacencySimplicialComplex(signed=False)
        result = transform(data)
        adj = result.adjacency_0.to_dense()
        # rank 0 adjacency: (n_vertices x n_vertices)
        assert adj.shape == (4, 4)

    def test_adjacency_zero_diagonal(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = AdjacencySimplicialComplex(signed=False)
        result = transform(data)
        adj = result.adjacency_0.to_dense()
        assert torch.allclose(torch.diag(adj), torch.zeros(4))


class TestCoadjacency:
    def test_coadjacency_shape(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = CoadjacencySimplicialComplex(signed=False)
        result = transform(data)
        coadj = result.coadjacency_1.to_dense()
        # rank 1 coadjacency: (n_edges x n_edges)
        assert coadj.shape == (6, 6)

    def test_coadjacency_zero_diagonal(self):
        data = _make_data(TETRAHEDRON_TRI)
        transform = CoadjacencySimplicialComplex(signed=False)
        result = transform(data)
        coadj = result.coadjacency_1.to_dense()
        assert torch.allclose(torch.diag(coadj), torch.zeros(6))
