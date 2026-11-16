from collections import defaultdict
from itertools import combinations
from typing import Optional

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx


def simplex_feature_index(top_simplices):
    """Map each simplex to its row in the per-rank feature tensors.

    Propagating transforms (cf. `PropagateConvexComb`) order each
    rank's feature tensor lexicographically over *all* simplices of
    that rank, except for rank 0, whose tensor is the raw vertex
    tensor indexed by (zero-based) vertex id. This mapping mirrors
    that ordering so features can be looked up per simplex.
    """
    m = len(top_simplices[0])
    simplices = set(top_simplices)
    for s in top_simplices:
        for dim in range(1, m):
            simplices.update(combinations(s, dim))

    by_rank = defaultdict(list)
    for s in simplices:
        by_rank[len(s)].append(s)

    index = {}
    for k, rank_simplices in by_rank.items():
        if k == 1:
            index.update((s, s[0] - 1) for s in rank_simplices)
        else:
            index.update((s, i) for i, s in enumerate(sorted(rank_simplices)))
    return index


class HasseDiagram(BaseTransform):
    def __init__(
        self,
        feature_propagation: Optional[str] = None,
        max_rank: Optional[int] = None,
        rank_feature: bool = False,
    ):
        """
        Parameters
        ----------
        feature_propagation : str, optional
            Name of the per-rank feature tensors `<name>_<rank>`, e.g.
            written by `PropagateConvexComb`; each node receives the
            row of its simplex, grouped into `x`.
        max_rank : int, optional
            Keep only the simplices of rank at most `max_rank` (rank is
            the number of vertices minus one) and the covering
            relations between them, so `max_rank=1` is the vertex-edge
            incidence graph. By default all ranks are kept.
        rank_feature : bool
            Append a one-hot encoding of the rank to each row of `x`,
            of width `max_rank + 1`, or top rank + 1 if `max_rank` is
            `None`. The propagated features need not identify the rank:
            the barycentre of an edge can coincide with a vertex.
            Ignored unless `feature_propagation` is set.
        """
        if max_rank is not None and max_rank < 0:
            raise ValueError(f"max_rank must be >= 0, got {max_rank}")
        self.feature_propagation = feature_propagation
        self.max_rank = max_rank
        self.rank_feature = rank_feature

    def forward(self, data: Data) -> Data:
        """Creates the Hasse diagram for a given triangulation.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Data containing information about a triangulation. This
            needs to at least include a `triangulation` key.

        Returns
        -------
        torch_geometric.data.Data
            Adjusted data object with all keys maintained, an
            `edge_index` tensor joining each simplex to the simplices
            of the next rank that contain it, and a boolean
            `vertex_mask` that is `True` for the vertex nodes. These
            come first and in vertex order, followed by the higher
            simplices rank by rank in lexicographic order, so
            node-level targets defined per vertex align with
            `x[vertex_mask]`. `simplex` lists the zero-based vertices
            of each node and `n_vertices` becomes the number of nodes.
        """
        top_simplices = sorted({tuple(s) for s in data["triangulation"]})
        top_simplices.sort(key=len)

        n_ranks = (
            len(top_simplices[-1])
            if self.max_rank is None
            else self.max_rank + 1
        )
        simplices = {
            face
            for top_simplex in top_simplices
            for k in range(1, n_ranks + 1)
            for face in combinations(top_simplex, k)
        }
        feat_index = (
            simplex_feature_index(top_simplices)
            if self.feature_propagation
            else None
        )

        G = nx.Graph()

        # `from_networkx` keeps insertion order, so vertex v becomes
        # node v - 1 and every facet exists before its cofaces.
        for simplex in sorted(simplices, key=lambda s: (len(s), s)):
            rank = len(simplex) - 1
            attrs = {
                "simplex": [v - 1 for v in simplex],
                "vertex_mask": rank == 0,
            }
            if self.feature_propagation:
                per_rank = getattr(data, f"{self.feature_propagation}_{rank}")
                features = per_rank[feat_index[simplex]]
                if self.rank_feature:
                    one_hot = torch.nn.functional.one_hot(
                        torch.tensor(rank), n_ranks
                    )
                    features = torch.cat([features, one_hot.to(features)])
                attrs[self.feature_propagation] = features
            G.add_node(simplex, **attrs)
            if rank > 0:
                G.add_edges_from(
                    (facet, simplex) for facet in combinations(simplex, rank)
                )

        if self.feature_propagation:
            data_ = from_networkx(
                G, group_node_attrs=[self.feature_propagation]
            )
        else:
            data_ = from_networkx(G)

        for k, v in data_.items():
            assert k not in data
            data[k] = v

        data["n_vertices"] = G.number_of_nodes()

        return data
