from typing import Optional

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx

from mantra.representations.graph.hasse_diagram import simplex_feature_index


class LeviGraph(BaseTransform):
    def __init__(self, feature_propagation: Optional[str] = None):
        super().__init__()
        self.feature_propagation = feature_propagation

    def forward(self, data: Data):
        """Retrieves the Levi Graph[1] of a triangulation
        if it's interpreted as a `configuration` between 0-simplices
        and maximal simplices.

        [1] Hauschild, J., Ortiz, J., & Vega, O. (2015). On the Levi graph of point-line configurations. Involve, a Journal of Mathematics, 8(5), 893-900.
        https://msp.org/involve/2015/8-5/involve-v8-n5-p14-s.pdf

        Parameters
        ----------
        data : torch_geometric.data.Data
            Data containing information about a triangulation. This
            needs to at least include a `triangulation` key.

        Returns
        -------
        torch_geometric.data.Data
            Adjusted data object with all keys maintained and an
            `edge_index` tensor for representing the Levi graph being
            present.
        """
        G = self._build_levi(data["triangulation"], data)

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

    def _build_levi(self, top_simplices, data):
        """Constructs the Levi graph.

        The Levi graph of a triangulation $T$ of dimension $d$ is a
        bipartite graph that has 0-simplices and maximal simplices as
        nodes and each 0-simplex is connected to a maximal simplex if
        it's contained in it.

        """
        # Guarantee the ordering
        top_simplices = list(set([tuple(s) for s in top_simplices]))
        top_simplices.sort()
        top_simplices.sort(key=len)

        feat_index = (
            simplex_feature_index(top_simplices)
            if self.feature_propagation
            else None
        )

        G = nx.Graph()

        # Collect all the vertices composing the simplices. Vertex
        # labels are contiguous and 1-indexed, so vertex $v$ becomes
        # node $v - 1$.
        vertices = sorted({v for simp in top_simplices for v in simp})
        n = len(vertices)

        # Add every 0-simplex as a node *first*, in sorted (ascending)
        # order, so that `from_networkx` maps them to consecutive
        # integers aligned with the original (zero-indexed) vertex
        # order.
        for v in vertices:
            attrs = {"simplex": [v - 1]}
            if self.feature_propagation:
                feat_tensor = getattr(data, f"{self.feature_propagation}_0")
                attrs[self.feature_propagation] = feat_tensor[v - 1]
            G.add_node(v - 1, **attrs)

        # For each maximal simplex
        for i, simp in enumerate(top_simplices):
            attrs = {"simplex": [v - 1 for v in simp]}
            if self.feature_propagation:
                feat_tensor = getattr(
                    data, f"{self.feature_propagation}_{len(simp) - 1}"
                )
                attrs[self.feature_propagation] = feat_tensor[feat_index[simp]]
            G.add_node(n + i, **attrs)
            # Connect the maximal simplex to the 0-simplices it contains
            for v in simp:
                G.add_edge(v - 1, n + i)

        return G
