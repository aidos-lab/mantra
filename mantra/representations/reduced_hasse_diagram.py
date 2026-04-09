import networkx as nx

from typing import Optional

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx


class ReducedHasseDiagram(BaseTransform):
    """Reduced Hasse diagram with only vertices and top-level simplices.

    Builds a bipartite graph connecting each vertex (0-simplex) to
    the top-level simplices it belongs to, skipping all intermediate
    faces.
    """

    def __init__(self, feature_propagation: Optional[str] = None):
        self.feature_propagation = feature_propagation

    def forward(self, data):
        """Creates the reduced Hasse diagram for a given triangulation.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Data containing information about a triangulation. This
            needs to at least include a `triangulation` key.

        Returns
        -------
        torch_geometric.data.Data
            Adjusted data object with all keys maintained and an
            `edge_index` tensor for representing the reduced Hasse
            diagram being present.
        """
        top_simplices = list(
            set([tuple(s) for s in data["triangulation"]])
        )
        top_simplices.sort()
        top_simplices.sort(key=len)

        G = self._build_reduced_hasse_diagram(top_simplices, data)
        group_node_attrs = (
            None
            if self.feature_propagation is None
            else [self.feature_propagation]
        )
        data_ = from_networkx(G, group_node_attrs=group_node_attrs)

        for k, v in data_.items():
            assert k not in data
            data[k] = v

        data["n_vertices"] = G.number_of_nodes()

        return data

    def _build_reduced_hasse_diagram(self, top_simplices, data):
        """Build a bipartite graph of vertices and top-level simplices.

        Vertices (0-simplices) are represented by their integer label.
        Top-level simplices are represented by their tuple. Edges
        connect each vertex to every top-simplex containing it.

        Parameters
        ----------
        top_simplices : list of tuples
            Top-level simplices as sorted tuples of 1-indexed vertices.
        data : torch_geometric.data.Data
            Original data object for feature propagation.

        Returns
        -------
        nx.Graph
            Bipartite graph with vertex and top-simplex nodes.
        """
        G = nx.Graph()

        # Collect all vertices
        all_vertices = sorted(
            set(v for s in top_simplices for v in s)
        )

        # Add vertex nodes (0-simplices)
        for v in all_vertices:
            extra_attr_dict = {"simplex": [v - 1]}
            if self.feature_propagation is not None:
                extra_attr_dict[self.feature_propagation] = data[
                    self.feature_propagation
                ][0][v - 1]
            G.add_node(v, **extra_attr_dict)

        # Add top-simplex nodes and connect to their vertices
        for i, top_simp in enumerate(top_simplices):
            extra_attr_dict = {
                "simplex": [sim - 1 for sim in top_simp]
            }
            if self.feature_propagation is not None:
                extra_attr_dict[self.feature_propagation] = data[
                    self.feature_propagation
                ][len(top_simp) - 1][i]
            G.add_node(top_simp, **extra_attr_dict)

            for v in top_simp:
                G.add_edge(v, top_simp)

        return G
