from collections import defaultdict
from itertools import combinations
from typing import Optional, Tuple

import networkx as nx
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx


class HasseDiagram(BaseTransform):
    def __init__(self, feature_propagation: Optional[str] = None):
        self.feature_propagation = feature_propagation

    def forward(self, data):
        """Creates the Hasse diagram for a given triangulation.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Data containing information about a triangulation. This
            needs to at least include a `triangulation` key.

        Returns
        -------
        torch_geometric.data.Data
            Adjusted data object with all keys maintained and an `edge_index`
            tensor for representing the Hasse diagram being present.
        """

        top_simplices = list(set([tuple(s) for s in data["triangulation"]]))

        top_simplices.sort()
        top_simplices.sort(key=len)

        feat_index = (
            self._feature_index(top_simplices)
            if self.feature_propagation
            else None
        )
        G = self._build_hasse_diagram(top_simplices, data, feat_index)

        if self.feature_propagation:
            data_ = from_networkx(
                G, group_node_attrs=[self.feature_propagation]
            )
        else:
            data_ = from_networkx(G)

        # Copy information from smaller `data_` object to the original
        # `data` tensor. This operates under the assumption that keys
        # are distinct.
        for k, v in data_.items():
            assert k not in data
            data[k] = v

        data["n_vertices"] = G.number_of_nodes()

        return data

    def _feature_index(self, top_simplices):
        """Map each simplex to its row in the per-rank feature tensors.

        Propagating transforms (cf. `_propagate_values`) order each
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
                index.update(
                    (s, i) for i, s in enumerate(sorted(rank_simplices))
                )
        return index

    def _build_connecting_lower_simplices(
        self, G: nx.Graph, data, k_simplex: Tuple[int], feat_index
    ) -> None:
        """
        Create the Hasse diagram layer corresponding to k_simplices.

        Given a graph and a k-simplex, add all k-1 simplices that are contained in
        the k-simplex in Hasse diagram G, then connect them with edges.

        Parameters
        ----------
        G : nx.Graph
            The `Graph` object used to construct the Hasse diagram.
        k_simplex : Tuple[Int]
            The tuple of vertices representing a k-simplex.
        Returns
        --------
        None

        """
        if len(k_simplex) == 1:
            return

        new_nodes = []
        k_minus_1_simplices = list(combinations(k_simplex, len(k_simplex) - 1))
        k_minus_1_simplices.sort()
        k_minus_1_simplices.sort(key=len)

        # For each k-1 simplex
        for k_simp in k_minus_1_simplices:
            k_simp = tuple(k_simp)
            new_nodes.append(k_simp)

            # A node reached via several parents only needs to be
            # built (and recursed into) once.
            if G.has_node(k_simp):
                continue

            extra_attr_dict = {"simplex": [sim - 1 for sim in k_simp]}

            if self.feature_propagation:
                vtx_feat_str = f"{self.feature_propagation}_{len(k_simp)-1}"
                feat_vtx_tensor = getattr(data, vtx_feat_str)
                extra_attr_dict[self.feature_propagation] = feat_vtx_tensor[
                    feat_index[k_simp]
                ]

            G.add_node(k_simp, **extra_attr_dict)

            self._build_connecting_lower_simplices(
                G, data, k_simp, feat_index
            )

        # TODO: What does edge_features look like in this case ?
        for new_node in new_nodes:
            G.add_edge(k_simplex, new_node)

    def _build_hasse_diagram(self, top_simplices, data, feat_index=None):
        """
        Construct the Hasse diagram out of the triangulation of a
        $d$-manifold. There is a vertex for each k-simplex and it's joined
        in a directed fashion to the k+1 simplex it is a subset of.

        Parameters
        ----------
        top_simplices : list of lists
            Top-level simplices specified as list of lists (or any other
            "iterable" data structure).

        Returns
        -------
        nx.Graph
            An directed graph describing Hasse diagram of a triangulation.
        """
        G = nx.Graph()

        for top_simp in top_simplices:
            extra_attr_dict = {"simplex": [sim - 1 for sim in top_simp]}

            if self.feature_propagation:
                vtx_feat_str = f"{self.feature_propagation}_{len(top_simp)-1}"
                feat_vtx_tensor = getattr(data, vtx_feat_str)
                extra_attr_dict[self.feature_propagation] = feat_vtx_tensor[
                    feat_index[top_simp]
                ]
            G.add_node(top_simp, **extra_attr_dict)
            self._build_connecting_lower_simplices(
                G, data, top_simp, feat_index
            )

        return G
