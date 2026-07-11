from collections import defaultdict
from itertools import combinations
from typing import Optional

import networkx as nx
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx


class DualGraph(BaseTransform):
    def __init__(self, feature_propagation: Optional[str] = None):
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
        assert "dimension" in data
        top_simplices = list(
            set([tuple(s) for s in data["triangulation"]])
        )  # Guarantee the ordering
        # Lexicographical sort
        top_simplices.sort()
        top_simplices.sort(key=len)

        G = self._build_dual_graph(data, top_simplices)

        # Here we assign the same name to nodes and edges
        if self.feature_propagation:
            data_ = from_networkx(
                G,
                group_node_attrs=[self.feature_propagation],
                group_edge_attrs=[self.feature_propagation],
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
            G.add_node(
                i, **extra_attr_dict
            )  # -1 to convert 1-index to 0-indexed

        # Remember which (d-1)-face induced each edge so features can
        # be propagated onto the edges below.
        edge_faces = {}

        # Add an edge to connect all cofaces. Notice that we implicitly only
        # ever consider valid cofaces, i.e., list of length at least two.
        for face, cofaces in face_to_cofaces.items():
            for a, b in combinations(sorted(cofaces), 2):
                G.add_edge(a, b)
                edge_faces[(a, b)] = face

        # Here we do an extra step to assign features to the simplices
        if self.feature_propagation:
            # Extract the correct strings for feature propagation
            feat_vtx_str = f"{self.feature_propagation}_{m-1}"
            feat_edge_str = f"{self.feature_propagation}_{m-2}"

            vtx_feat_tensor = getattr(data, feat_vtx_str)
            edge_feat_tensor = getattr(data, feat_edge_str)

            vtx_feat_dict = {
                i: vtx_feat_tensor[i] for i in range(vtx_feat_tensor.shape[0])
            }

            # The per-rank feature tensors are ordered lexicographically
            # over all simplices of the rank; look each edge's inducing
            # face up in that ordering.
            face_index = {
                f: i for i, f in enumerate(sorted(face_to_cofaces))
            }
            edge_feat_dict = {
                t: edge_feat_tensor[face_index[f]]
                for t, f in edge_faces.items()
            }

            # Here we just set it
            nx.set_node_attributes(
                G, values=vtx_feat_dict, name=self.feature_propagation
            )
            nx.set_edge_attributes(
                G, values=edge_feat_dict, name=self.feature_propagation
            )
        return G
