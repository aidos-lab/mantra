from abc import abstractmethod, ABCMeta
from typing import Dict, List

from networkx import incidence_matrix
import numpy as np

from scipy.sparse import csr_matrix
import scipy

import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from mantra.representations.simplex_trie import SimplexTrie
from mantra.representations.simplex_trie import Simplex


class AddSimplexTrie(BaseTransform):
    """Add simplex trie to object.

    Takes in a `Data` object and computes a `SimplexTrie` for it.
    Then it adds it to it's storage to save up on computation
    of connectivity matrices.
    """

    def forward(self, data: Data):
        assert "triangulation" in data
        simplex_trie = SimplexTrie()
        for t in data.triangulation:
            simplex_trie.insert(t)
        data.simplex_trie = simplex_trie
        return data


class AbstractSimplicialComplexConnectivity(BaseTransform, ABCMeta):
    """Base class for connectivity transforms.

    Parent class for implementing a transform that adds a
    connectivity matrix for simplices. This transform takes a `Data` object with a `triangulation`
    field and represents this triangulation by the canonical
    representation of it's neighborhood relationships in the
    PyG edge-index format.
    """

    def __init__(self, signed: bool, connectivity_name: str, index: bool):
        self.signed = signed
        self.connectivity_name = connectivity_name
        self.index = index

    def skeleton(self, simplex_trie: SimplexTrie, rank: int) -> List[Simplex]:
        """Compute the r-skeleton of a simplex.

        Parameters
        ----------
        simplex_trie: SimplexTrie
            The datastructure containing a simplex.
        rank: int
            Rank r of the skeleton.
        Returns:
            List[Simplex]

        """
        return sorted(node.simplex for node in simplex_trie.skeleton(rank))

    @abstractmethod
    def generate_matrix(
        self, simplex_trie: SimplexTrie, rank: int, max_rank: int
    ) -> scipy.sparse.csr_matrix:
        """Generate a connectivity matrix.

        Parameters
        ----------
        simplex_trie: SimplexTrie
            The datastructure contating the simplicial complex.
        rank: int
            The rank r for which to generate the connectivity relationship.
        max_rank: int
            Maximum rank of the simplex trie.

        Returns
        -------
            Torch.tensor (torch.sparse.coo)
        """
        return

    def forward(self, data: Data):
        # If the trie is not already calculated then this calculates it and adds it
        # to the data object
        if "simplex_trie" not in data:
            assert (
                "triangulation" in data
            ), "Your data object does not contain a triangulation or a simplex trie"
            data = AddSimplexTrie()(data)

        max_rank = data.dimension.item()
        # The shape that empty tensors should take
        shape = list(
            np.pad(
                list(data.simplex_trie.shape),
                (0, max_rank + 1 - len(data.simplex_trie.shape)),
            )
        )

        # NOTE: For some connectivity relatioship there will be 0 matrices
        for rank_idx in range(0, max_rank + 1):
            connectivity_name = f"{self.connectivity_name}_{rank_idx}"
            try:
                data[connectivity_name] = _from_sparse(
                    self.generate_matrix(data.simplex_trie, rank_idx, max_rank),
                    device=data.triangulation.device
                )
            except ValueError as e:
                idx_low_simp = rank_idx - 1 if rank_idx > 0 else rank_idx
                if "incidence" in self.connectivity_name:
                    data[connectivity_name] = torch.zeros(
                        [shape[idx_low_simp], shape[rank_idx]],
                        layout=torch.sparse_coo,
                    ).coalesce().to(data.triangulations.device)
                elif "adjacency" in self.connectivity_name:
                    data[connectivity_name] = torch.zeros(
                        [shape[rank_idx], shape[rank_idx]],
                        layout=torch.sparse_coo,
                    ).coalesce().to(data.triangulations.device)

        return data


class IncidenceSimplicialComplex(AbstractSimplicialComplexConnectivity):
    """Add incidences of a simplicial complex."""

    def __init__(self, signed: bool, index=False):
        super().__init__(signed, "incidence", index)

    def generate_matrix(
        self, simplex_trie: SimplexTrie, rank: int, max_rank: int
    ) -> scipy.sparse.csr_matrix:
        idx_simplices, idx_faces, values = [], [], []

        simplex_dict_d = {
            tuple(sorted(simplex)): i
            for i, simplex in enumerate(self.skeleton(simplex_trie, rank))
        }
        simplex_dict_d_minus_1 = {
            tuple(sorted(simplex)): i
            for i, simplex in enumerate(self.skeleton(simplex_trie, rank - 1))
        }
        for simplex, idx_simplex in simplex_dict_d.items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1) ** i)
                face = frozenset(simplex).difference({left_out})
                idx_faces.append(simplex_dict_d_minus_1[tuple(sorted(face))])

        if len(values) != (rank + 1) * len(simplex_dict_d):
            raise RuntimeError("Error in computing the incidence matrix.")

        boundary = csr_matrix(
            (values, (idx_faces, idx_simplices)),
            dtype=np.float32,
            shape=(
                len(simplex_dict_d_minus_1),
                len(simplex_dict_d),
            ),
        )

        if not self.signed:
            boundary = abs(boundary)
        if self.index:
            return simplex_dict_d_minus_1, simplex_dict_d, boundary
        return boundary


class UpLaplacianSimplicialComplex(AbstractSimplicialComplexConnectivity):
    """Add Up Laplacian of a simplicial complex."""

    def __init__(self, signed: bool, index: bool = False):
        super().__init__(signed, "up_laplacian", index=index)

    def generate_matrix(
        self, simplex_trie: SimplexTrie, rank: int, max_rank: int
    ):
        incidence_matrix_transform = IncidenceSimplicialComplex(
            self.signed, index=True
        )

        if rank < max_rank and rank >= 0:
            row, _, B_next = incidence_matrix_transform.generate_matrix(
                simplex_trie, rank + 1, max_rank
            )
            L_up = B_next @ B_next.transpose()
        else:
            raise ValueError(
                f"Rank should larger than 0 and <= {max_rank - 1} (maximal dimension cells-1), got {rank}"
            )
        if not self.signed:
            L_up = abs(L_up)
        if self.index:
            return row, L_up

        return L_up


class DownLaplacianSimplicialComplex(AbstractSimplicialComplexConnectivity):
    """Add Down Laplacian of a simplicial complex."""

    def __init__(self, signed: bool, index: bool = False):
        super().__init__(signed, "down_laplacian", index=index)

    def generate_matrix(
        self, simplex_trie: SimplexTrie, rank: int, max_rank: int
    ):
        incidence_matrix_transform = IncidenceSimplicialComplex(
            self.signed, index=True
        )

        if max_rank >= rank > 0:
            _, column, B = incidence_matrix_transform.generate_matrix(
                simplex_trie, rank, max_rank
            )
            L_down = B.transpose() @ B
        else:
            raise ValueError(
                f"Rank should be larger than 1 and <= {max_rank} (maximal dimension cells), got {rank}."
            )
        if not self.signed:
            L_down = abs(L_down)
        if self.index:
            return column, L_down
        return L_down


class AdjacencySimplicialComplex(AbstractSimplicialComplexConnectivity):
    """Add adjacencies of a simplicial complex."""

    def __init__(self, signed: bool):
        super().__init__(signed, "adjacency", index=False)

    def generate_matrix(
        self, simplex_trie: SimplexTrie, rank: int, max_rank: int
    ):
        up_lap_transform = UpLaplacianSimplicialComplex(self.signed, index=True)
        ind, l_up = up_lap_transform.generate_matrix(
            simplex_trie, rank, max_rank
        )

        l_up.setdiag(0)

        if not self.signed:
            l_up = abs(l_up)
        return l_up


class CoadjacencySimplicialComplex(AbstractSimplicialComplexConnectivity):
    """Add coadjacencies of a simplicial complex.

        Notes
        -----
        This constructs matrices that relate two simplices $\\sigma, $\\tau$
        if there is a $\\lambda$ whose rank is lower than both $\\sigma,\\tau$
        and $\\lambda \\subset \\sigma \\wedge \\lambda \\subset \\tau$.

        Example
        -------
        triangulations = [
            [0, 1, 2],
            [1, 2, 3]
        ]
        data = Data(triangulation = triangulation)
        transform = CoadjacencySimplicialComplex(signed=False)
        data = transform(data)
        # Then the tensors look like
        data.coadjacency_0 = torch.zeros(4,4)
        data.coadjacency_1 = [ # (0, 1), (0, 2), (1, 2), (1, 3), (2, 3)
            [0, 1, 1, 1, 1]
            [1, 0, 1, 0, 1]
            [1, 1, 0, 1, 1]
            [1, 0, 1, 0, 1]
            [0, 1, 1, 1, 0]
        ]
        data.coadjacency_2 = [ # (0, 1, 2), (1, 2, 3)
            [0, 1]
            [1, 0]
        ]
    """

    def __init__(self, signed: bool):
        super().__init__(signed, "coadjacency", index=False)

    def generate_matrix(
        self, simplex_trie: SimplexTrie, rank: int, max_rank: int
    ):
        down_lap_transform = DownLaplacianSimplicialComplex(self.signed, index=True)

        ind, L_down = down_lap_transform.generate_matrix(
            simplex_trie, rank, max_rank
        )
        L_down.setdiag(0)
        if not self.signed:
            L_down = abs(L_down)
        return L_down


def _from_sparse(data: scipy.sparse.csc_matrix, device=None) -> torch.Tensor:
    """Convert sparse input data directly to torch sparse coo format.

    Parameters
    ----------
    device: Torch device where we want to store the data.
    data : scipy.sparse._csc.csc_matrix
        Input n_dimensional data.

    Returns
    -------
    torch.sparse_coo, same shape as data
        input data converted to tensor.
    """
    if device is None:
        device = data.device("cpu")
    # cast from csc_matrix to coo format for compatibility
    coo = data.tocoo()

    values = torch.FloatTensor(coo.data, device=device)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)), device=device)
    sparse_data = torch.sparse_coo_tensor(
        indices, values, coo.shape,
        device=device
    ).coalesce()
    return sparse_data
