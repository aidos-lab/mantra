import networkx as nx

from collections import defaultdict
from itertools import combinations

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx


class DualGraph(BaseTransform):
    def __init__(self, feature_propagation: str | None = None):
        self.feature_propagation = feature_propagation

    def forward(self, data):
        """Creates dual graph for a given triangulation.

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
        top_simplices = list(
            set([tuple(s) for s in data["triangulation"]])
        )  # Guarantee the ordering
        # Lexicographical sort
        top_simplices.sort()
        top_simplices.sort(key=len)

        G = self._build_dual_graph(data, top_simplices)
        node_attributes_to_keep = []  # ["simplex"]

        if self.feature_propagation is not None:
            node_attributes_to_keep.append(self.feature_propagation)

        data_ = from_networkx(G, group_node_attrs=node_attributes_to_keep)

        # Copy information from smaller `data_` object to the original
        # `data` tensor. This operates under the assumption that keys
        # are distinct.
        for k, v in data_.items():
            assert k not in data
            data[k] = v

        data["n_vertices"] = G.number_of_nodes()
        print(data)
        return data

    def _build_dual_graph(self, data, top_simplices):
        """
        Construct the dual graph, i.e., the adjacency graph, of a triangulated
        $d$-manifold. The graph will have a vertex for each top-level simplex,
        and an edge if two such simplices are adjacent.

        Parameters
        ----------
        top_simplices : list of lists
            Top-level simplices specified as list of lists (or any other
            "iterable" data structure).

        Returns
        -------
        nx.Graph
            An undirected graph describing the dual graph of the
            triangulation. A vertex attribute (`simplex`) stores
            the original simplex corresponding to it.
        """
        G = nx.Graph()
        m = len(top_simplices[0])

        # Map each (d-1)-face to the list of incident top-simplices (by index)
        # First, we map each $(d-1)$-face to the list of its cofaces.
        face_to_cofaces = defaultdict(list)
        for i, s in enumerate(top_simplices):
            for face in combinations(s, m - 1):
                face_to_cofaces[face].append(i)

        # Every node in the graph corresponds to a top-level simplex.
        for i, s in enumerate(top_simplices):
            extra_attr_dict = {"simplex": [sim - 1 for sim in s]}
            if self.feature_propagation is not None:
                extra_attr_dict[self.feature_propagation] = data[
                    self.feature_propagation
                ][len(s) - 1][i]
            G.add_node(
                i, **extra_attr_dict
            )  # -1 to convert 1-index to 0-indexed

        # Add an edge to connect all cofaces. Notice that we implicitly only
        # ever consider valid cofaces, i.e., list of length at least two.
        for face, cofaces in face_to_cofaces.items():
            for a, b in combinations(sorted(cofaces), 2):
                G.add_edge(a, b)

        return G
