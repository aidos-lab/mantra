import networkx as nx
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx


class LeviGraph(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data: Data):
        """Retrieves the Levi Graph[1] of a triangulation
        if it's interpreted as a `configuration` between 0-simplices
        and maximal simplices.

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

        data["n_vertices"] = G.number_of_nodes()
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

        # Number of maximal simplices
        m = len(top_simplices)


        # Collect all the nodes
        # composing the simplices
        for top_simp in top_simplices:
            nodes.update(tuple(top_simp))

        n = len(nodes)

        # We add a node for all the pre-existing nodes
        for i, node in enumerate(list(nodes)):
            G.add_node(i, simplex=[i])

        # For each maximal simplex
        for i, simp in enumerate(top_simplices):
            G.add_node(n+i, simplex=simp)
            # Add edges
            for node in simp:
                G.add_edge(node-1, n+i)

        return G
