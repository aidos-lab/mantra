from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, get_laplacian, to_dense_adj

import torch

# Small constant to avoid division by zero
EPSILON = 1e-6


def _get_fiedler_vector(data, **kwargs):
    """Calculate stable version of the Fiedler vector."""
    edge_index, edge_weight = get_laplacian(
        data.edge_index, normalization=None, num_nodes=data.num_nodes
    )

    L = to_dense_adj(
        edge_index, edge_attr=edge_weight, max_num_nodes=data.num_nodes
    ).squeeze(0)

    _, V = torch.linalg.eigh(L)

    fiedler = V[:, 1]
    fiedler = fiedler / (fiedler.abs().max() + EPSILON)

    # Make us invariant with respect to sign flips by ensuring the
    # element with maximum absolute value is positive.
    s = torch.sign(fiedler[torch.argmax(fiedler.abs())])
    fiedler = fiedler * s

    return fiedler


class EquivariantMomentCurveEncoding(BaseTransform):
    def __init__(
        self,
        order=8,
        store_edge_attr=True,
        deg_attr_name: str = "x_deg_enc",
        attr_name_cheby: str = "x_moment_enc",
        edge_attr_name_cheby: str = "edge_attr_moment_enc",
        attr_name_pow: str = "x_moment_enc_pow",
        edge_attr_name_pow: str = "edge_attr_moment_enc_pow",
        use_cheby_basis: bool = True,
    ):
        self.order = order
        self.store_edge_attr = store_edge_attr
        self.deg_attr_name = deg_attr_name
        self.attr_name_cheby = attr_name_cheby
        self.edge_attr_name_cheby = edge_attr_name_cheby
        self.attr_name_pow = attr_name_pow
        self.edge_attr_name_pow = edge_attr_name_pow
        self.use_cheby_basis = use_cheby_basis

    def forward(self, data):
        x = _get_fiedler_vector(data)

        # Chebyshev basis
        T_prev = torch.ones_like(x)
        T_curr = x
        cheby_features = [T_curr]
        for _ in range(self.order - 1):
            T_next = 2 * x * T_curr - T_prev
            cheby_features.append(T_next)
            T_prev = T_curr
            T_curr = T_next
        setattr(data, self.attr_name_cheby, torch.stack(cheby_features, dim=1))

        # Power basis
        powers = [x]
        for k in range(2, self.order + 1):
            powers.append(x**k)
        setattr(data, self.attr_name_pow, torch.stack(powers, dim=1))

        if self.deg_attr_name:
            deg = degree(
                data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float32
            ).unsqueeze(-1)
            setattr(data, self.deg_attr_name, deg)

        if self.store_edge_attr and data.edge_index is not None:
            source, target = data.edge_index
            x_moment_cheby = getattr(data, self.attr_name_cheby)
            x_moment_pow = getattr(data, self.attr_name_pow)
            edge_attr_cheby = (x_moment_cheby[source] + x_moment_cheby[target]) / 2.0
            edge_attr_pow = (x_moment_pow[source] + x_moment_pow[target]) / 2.0
            setattr(data, self.edge_attr_name_cheby, edge_attr_cheby)
            setattr(data, self.edge_attr_name_pow, edge_attr_pow)

        return data


class UseMomentCurveFeatures(BaseTransform):
    def __init__(
        self,
        cat=False,
        use_edge_attr=True,
        include_degree: bool = False,
        deg_attr_name: str = "x_deg_enc",
        source_attr_cheby: str = "x_moment_enc",
        edge_attr_name_cheby: str = "edge_attr_moment_enc",
        source_attr_pow: str = "x_moment_enc_pow",
        edge_attr_name_pow: str = "edge_attr_moment_enc_pow",
        use_cheby_basis: bool = True,
    ):
        self.cat = cat
        self.use_edge_attr = use_edge_attr
        self.include_degree = include_degree
        self.deg_attr_name = deg_attr_name
        self.source_attr_cheby = source_attr_cheby
        self.edge_attr_name_cheby = edge_attr_name_cheby
        self.source_attr_pow = source_attr_pow
        self.edge_attr_name_pow = edge_attr_name_pow
        self.use_cheby_basis = use_cheby_basis

    def forward(self, data):
        source_attr = (
            self.source_attr_cheby if self.use_cheby_basis else self.source_attr_pow
        )
        edge_attr_name = (
            self.edge_attr_name_cheby
            if self.use_edge_attr and self.use_cheby_basis
            else self.edge_attr_name_pow
        )

        if not hasattr(data, source_attr) or getattr(data, source_attr) is None:
            raise RuntimeError(
                "Moment curve features missing. Re-run data processing with --force_reload to cache them."
            )

        x_moment = getattr(data, source_attr)

        if self.include_degree:
            if not hasattr(data, self.deg_attr_name):
                raise RuntimeError(
                    f"Missing degree attribute '{self.deg_attr_name}'. Re-run preprocessing with MomentCurveEncoding."
                )
            deg = getattr(data, self.deg_attr_name)
            x_moment = torch.cat([deg, x_moment], dim=-1)

        if data.x is None or not self.cat:
            data.x = x_moment
        else:
            data.x = torch.cat([data.x, x_moment], dim=-1)

        if self.use_edge_attr and hasattr(data, edge_attr_name):
            edge_attr_moment = getattr(data, edge_attr_name)
            if data.edge_attr is None or not self.cat:
                data.edge_attr = edge_attr_moment
            else:
                data.edge_attr = torch.cat([data.edge_attr, edge_attr_moment], dim=-1)

        return data


if __name__ == "__main__":
    from torch_geometric.data import Data

    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            [2, 5, 2, 3, 4, 0, 1, 5, 1, 6, 1, 6, 0, 2, 3, 4],
        ],
        dtype=torch.long,
    )

    node_attr = torch.tensor([[0], [1], [2], [3], [4], [5], [6]], dtype=torch.float)

    data = Data(num_nodes=7, x=node_attr, edge_index=edge_index)

    pre_transform = MomentCurveEncoding(order=2)
    runtime_transform = UseMomentCurveFeatures(cat=True)

    data = pre_transform(data)
    data = runtime_transform(data)

    print("Node Features (Moment Curve points):")
    print(data.x.shape)
    print(data.x)

    if "edge_attr" in data:
        print("Edge Features (Barycenters of 1-simplices):")
        print(data.edge_attr.shape)
        print(data.edge_attr)

