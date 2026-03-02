from torch_geometric.transforms import BaseTransform


import numpy as np
import torch


from typing import Tuple


def weighted_chain_laplacian(
    B_p, B_p_plus_1, W_p, W_p_plus_1, W_p_minus_1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcultes the weighted Laplacian of a p-chain.

    Parameters
    ---------
    B_p : np.ndarray [n_p-1, n_p]
        Boundary matrix of p-chains [number of  (p-1)-simplices, number of p-simplices.
    B_p_plus_1 :  np.ndarray [n_p, n_p+1]
        Boundary matrix of (p+1)-chains [number of  p-simplices, number of (p+1)-simplices.
    W_p : np.ndarray [n_p, n_p]
        Weight matrix of p-chains [number of  p-simplices, number of p-simplices.
    W_p_plus_1 : np.ndarray [n_p+1, n_p+1]
        Weight matrix of (p+1)-chains [number of  (p+1)-simplices, number of (p+1)-simplices.
    W_p_minus_1 : np.ndarray [n_p-1, n_p-1]
        Weight matrix of (p+1)-chains [number of  (p-1)-simplices, number of (p-1)-simplices.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of matrices:  1. Up Laplacian of the chain (L_p^{up}),
        2. Down Laplacian of the chain (L_{p}^{down}) and
        3. Hodge Laplacian of the chain (L_{p}^{hodge}),
    """

    inv_W_p = np.linalg.inv(W_p)
    inv_W_p_minus_1 = np.linalg.inv(W_p_minus_1)

    # Up Laplacian: B_{p+1} W_p B_{p+1}.T W_p^{-1}
    L_up: np.ndarray = B_p_plus_1 @ W_p_plus_1 @ B_p_plus_1.T @ inv_W_p

    # B_p.T W_{p-1}^{-1} B_p
    L_down: np.ndarray = W_p @ B_p.T @ inv_W_p_minus_1 @ B_p

    # L_hodge = L_up + L_down
    L_hodge: np.ndarray = L_up + L_down

    return (L_up, L_down, L_hodge)


def calculate_er(B_p, W_p_minus_1, L_up_p_minus_1):
    """ "
    Parameters
    ---------
    B_p : np.ndarray [n_p-1, n_p]
        Boundary matrix of p-chains [number of  (p-1)-simplices, number of p-simplices.
    W_p_minus_1 : np.ndarray [n_p-1, n_p-1]
        Weight matrix of (p+1)-chains [number of  (p-1)-simplices, number of (p-1)-simplices.
    L_up_p_minus_1 : np.ndarray [n_p-1, n_p-1]
        Up Laplacian matrix of the (p-1)-chain, denoted L_{p-1}^{up}.

    Returns
    -------
    np.ndarray [n_p]
        Effective vector resistance of the p-chain.
    """

    # W_p^{1/2}
    W_p_minus_1_sqrt = torch.sqrt(W_p_minus_1)

    # W_{p_1}^{-1/2}
    W_p_minus_1_sqrt_inv = torch.linalg.inv(W_p_minus_1_sqrt)

    # W_{p-1}^{-1/2} L_{p-1}^{up} W_{p-1}^{1/2}
    partial = W_p_minus_1_sqrt_inv @ L_up_p_minus_1 @ W_p_minus_1_sqrt

    # Moore-Penrose pseudoinv
    # NOTE: Look into the pseudoinverse precision (machine precision guarantees)
    pseudo_inv = np.linalg.pinv(partial)

    # B_p.T W_{p-1}^{-1/2} ( W_{p-1}^{-1/2} L_{p-1}^{up} W_{p-1}^{1/2} )^{*} W_{p-1}^{-1/2} B_p
    R_p = (
        B_p.T @ W_p_minus_1_sqrt_inv @ pseudo_inv @ W_p_minus_1_sqrt_inv @ B_p
    )

    return torch.diag(R_p)


class EffectiveResistanceEmbedding(BaseTransform):

    def __init__(self):
        """Create new moment curve embedding transform.

        Parameters
        ----------
        """
        super().__init__()

    def forward(self, data):
        """Calculate moment curve embedding for a data object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object

        Returns
        -------
        torch_geometric.data.Data
            Data object with a new `moment_curve_embedding` key added.
            The attribute will be overwritten if already present.
        """
        assert "n_vertices" in data and "dimension" in data

        d = data["dimension"]

        X = dict()

        for p in range(0, d):
            if p == 0:
                B_p_plus_1 = getattr(data, f"incidence_{p+1}")  # [n_p, n_p+1]
                B_p = torch.zeros((1, B_p_plus_1.shape[0]))  # [1, n_p]
                W_p_minus_1 = torch.ones((1, 1))  # [1,1]
                W_p = torch.eye(B_p.shape[1])  # [n_p, n_p]
                W_p_plus_1 = torch.eye(B_p_plus_1.shape[1])  # [ n_p+1, n_p+1]
            else:
                B_p_plus_1 = getattr(data, f"incidence_{p+1}")  # [n_p. n_p+1]
                B_p = getattr(data, f"incidence_{p}")  # [n_p-1, n_p]
                W_p_minus_1 = torch.eye(B_p.shape[0])  # [n_p-1, n_p-1]
                W_p = torch.eye(B_p.shape[1])  # [n_p, n_p]
                W_p_plus_1 = torch.eye(B_p_plus_1.shape[1])  # [n_p + 1, n_p+1]

            L_up_p, L_down_p, L_hodge_p = weighted_chain_laplacian(
                B_p, B_p_plus_1, W_p, W_p_plus_1, W_p_minus_1
            )

            R_p_plus_1 = calculate_er(B_p_plus_1, W_p, L_up_p)

            X[p + 1] = R_p_plus_1

        data.er = X

        return data


def er_statistics(x: torch.Tensor) -> torch.Tensor:
    """
    x: 1D tensor of effective resistances, shape (N,)
    returns: 1D tensor of statistics, shape (7,)
    """
    # Ensure float tensor
    x = x.float()

    mean = x.mean()
    std = x.std(unbiased=False)

    # Clamp tiny numerical noise
    eps = torch.finfo(x.dtype).eps
    tol = 10 * eps  # small multiple of machine precision
    std = torch.where(std.abs() < tol, torch.zeros_like(std), std)

    minv = x.min()
    maxv = x.max()

    # Quantiles
    q25 = torch.quantile(x, 0.25)
    median = torch.quantile(x, 0.50)
    q75 = torch.quantile(x, 0.75)

    return torch.stack([mean, std, minv, maxv, median, q25, q75])


class EffectiveResistanceStatisticsEmbedding(BaseTransform):

    def __init__(self):
        """Create new moment curve embedding transform.

        Parameters
        ----------
        """
        super().__init__()

    def forward(self, data):
        """Calculate effective resistance statistics for a data object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object.

        Returns
        -------
        torch_geometric.data.Data
            The same data object with a new attribute `er_stats`: torch.Tensor of shape (d, 7) added, d the dimension of the triangulation.
            Each row contains the seven summary statistics
            mean, standard deviation, min, max, median, 25% quantile, and 75% quantile
            of the effective resistances for each dimension 0, 1, ..., d-1.
            If `er_stats` already exists, it is overwritten.
        """
        assert "n_vertices" in data and "dimension" in data

        d = data["dimension"]

        # X = dict()
        stats = torch.empty(d, 7)

        for p in range(0, d):
            if p == 0:
                B_p_plus_1 = getattr(data, f"incidence_{p+1}")  # [n_p, n_p+1]
                B_p = torch.zeros((1, B_p_plus_1.shape[0]))  # [1, n_p]
                W_p_minus_1 = torch.ones((1, 1))  # [1,1]
                W_p = torch.eye(B_p.shape[1])  # [n_p, n_p]
                W_p_plus_1 = torch.eye(B_p_plus_1.shape[1])  # [ n_p+1, n_p+1]
            else:
                B_p_plus_1 = getattr(data, f"incidence_{p+1}")  # [n_p. n_p+1]
                B_p = getattr(data, f"incidence_{p}")  # [n_p-1, n_p]
                W_p_minus_1 = torch.eye(B_p.shape[0])  # [n_p-1, n_p-1]
                W_p = torch.eye(B_p.shape[1])  # [n_p, n_p]
                W_p_plus_1 = torch.eye(B_p_plus_1.shape[1])  # [n_p + 1, n_p+1]

            L_up_p, L_down_p, L_hodge_p = weighted_chain_laplacian(
                B_p, B_p_plus_1, W_p, W_p_plus_1, W_p_minus_1
            )

            R_p_plus_1 = calculate_er(B_p_plus_1, W_p, L_up_p)

            stats[p] = er_statistics(R_p_plus_1)

        data.er_stats = stats.flatten()

        return data
