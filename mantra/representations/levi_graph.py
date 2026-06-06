import networkx as nx
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx


class LeviGraph(BaseTransform):
    def forward(self, data):
        """Retrieves the Levi Graph[1] of a triangulation
        if it's interpreted as a `configuration`.

        [1] Hauschild, J., Ortiz, J., & Vega, O. (2015). On the Levi graph of point-line configurations. Involve, a Journal of Mathematics, 8(5), 893-900. 
        https://msp.org/involve/2015/8-5/involve-v8-n5-p14-s.pdf 

        .

        Parameters
        ----------
        data : torch_geometric.data.Data
            Data containing information about a triangulation. This
            needs to at least include a `triangulation` key.

        Returns
        -------
        torch_geometric.data.Data
            Adjusted data object with the `triangulation` key removed,
            all other keys maintained, and `edge_index` information of
            the Levy Graph being present.
        """
        G = self._build_levi(data["triangulation"])
        data_ = from_networkx(G)

        for k, v in data_.items():
            assert k not in data
            data[k] = v

        return data

    def _build_levi(self, top_simplices):
        """Constructs the Levy Graph.

        The Levi Graph of a triangulation $T$ of dimension $d$ is a graph
        that has 0-simplices and k-simplices as
        nodes and each 0-simplex is connected to a k-simplex if 
        it's contained in it.

        """
        G = nx.Graph()
        nodes = set()
        m = len(top_simplices)

        for simp in top_simplices:
            nodes = nodes.union(set(simp))

        n = len(nodes)

        for i, node in enumerate(list(nodes)):
            G.add_node(i)


        for i, simp in enumerate(top_simplices):
            G.add_node(n+i, simplex=simp)
            # Add edges
            for node in simp:
                G.add_edge(node-1, n+i)
        return G
