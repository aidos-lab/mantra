import networkx as nx

from collections import defaultdict
from itertools import combinations

from typing import List, Tuple 

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx


class HasseDiagram(BaseTransform):
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

        G = self._build_hasse_diagram(data["triangulation"])
        data_ = from_networkx(G)

        # Copy information from smaller `data_` object to the original
        # `data` tensor. This operates under the assumption that keys
        # are distinct.
        for k, v in data_.items():
            assert k not in data
            data[k] = v

        return data

    def _build_connecting_lower_simplices(self, G: nx.Graph, k_simplex: Tuple[int]) -> None:
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
        for k_simp in combinations(k_simplex, len(k_simplex)-1):

            k_simp = tuple(k_simp)

            G.add_node(k_simp)
            new_nodes.append(k_simp)

            self._build_connecting_lower_simplices(G, k_simp)

        for new_node in new_nodes:
            G.add_edge(k_simplex, new_node)


    def _build_hasse_diagram(self, top_simplices):

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
            G.add_node(top_simp)
            self._build_connecting_lower_simplices(G, top_simp)

        return G
