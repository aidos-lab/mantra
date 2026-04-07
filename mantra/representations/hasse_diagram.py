import networkx as nx

from itertools import combinations

from typing import Tuple, List, Optional

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
            tensor for representing the dual graph being present.
        """

        top_simplices = list(set([tuple(s) for s in data["triangulation"]]))

        top_simplices.sort()
        top_simplices.sort(key=len)

        G = self._build_hasse_diagram(top_simplices, data)
        group_node_attrs: List[str] = self.feature_propagation if self.feature_propagation is None else [self.feature_propagation]
        data_ = from_networkx(G, group_node_attrs=group_node_attrs)

        # Copy information from smaller `data_` object to the original
        # `data` tensor. This operates under the assumption that keys
        # are distinct.
        for k, v in data_.items():
            assert k not in data
            data[k] = v

        data["n_vertices"] = G.number_of_nodes()

        return data

    def _build_connecting_lower_simplices(
        self, G: nx.Graph, data, k_simplex: Tuple[int]
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
        Returuns
        --------
        None

        """
        if len(k_simplex) == 1:
            return

        new_nodes = []
        k_minus_1_simplices = list(combinations(k_simplex, len(k_simplex) - 1))
        k_minus_1_simplices.sort()
        k_minus_1_simplices.sort(key=len)

        for i, k_simp in enumerate(k_minus_1_simplices):

            k_simp = tuple(k_simp)
            extra_attr_dict = {"simplex": [sim - 1 for sim in k_simp]}
            if self.feature_propagation is not None:
                extra_attr_dict[self.feature_propagation] = data[
                    self.feature_propagation
                ][len(k_simp) - 1][i]
            G.add_node(k_simp, **extra_attr_dict)
            new_nodes.append(k_simp)

            self._build_connecting_lower_simplices(G, data, k_simp)

        for new_node in new_nodes:
            G.add_edge(k_simplex, new_node)

    def _build_hasse_diagram(self, top_simplices, data):
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

        for i, top_simp in enumerate(top_simplices):
            extra_attr_dict = {"simplex": [sim - 1 for sim in top_simp]}
            if self.feature_propagation is not None:
                extra_attr_dict[self.feature_propagation] = data[
                    self.feature_propagation
                ][len(top_simp) - 1][i]
            G.add_node(top_simp, **extra_attr_dict)
            self._build_connecting_lower_simplices(G, data, top_simp)

        return G
