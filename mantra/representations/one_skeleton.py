import networkx as nx


from mantra.representations.internal import SimplexTrie

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx


class OneSkeleton(BaseTransform):
    def forward(self, data):
        """Retrieves the 1-skeleton of a triangulation.

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
            the 1-skeleton being present.
        """
        G = self._build_one_skeleton(data["triangulation"])
        data_ = from_networkx(G)

        for k, v in data_.items():
            assert k not in data
            data[k] = v

        return data

    def _build_one_skeleton(self, top_simplices):
        """Constructs the 1-skeleton.

        The 1-skeleton of a triangulation $T$ is a graph
        that has 0-simplices as nodes and 1-simplices as
        edges.

        """
        # First we construct the Trie to optimally extract the 1-skeleton
        simplex_trie = SimplexTrie()
        for s in top_simplices:
            simplex_trie.insert(s)

        one_simplices = sorted(
            node.simplex for node in simplex_trie.skeleton(1)
        )

        G = nx.Graph()

        # We loop over each pair of 0-simplices (u,v), where s = (u,v)  is a 1-simplex
        for s in one_simplices:
            # These are the 0-simplices composing a 1-simplex
            u, v = list(s)

            # Add the nodes that are not contained in the graph yet
            for w in [u, v]:
                if w - 1 not in G:  # Convert to 0-index
                    G.add_node(w - 1)
            G.add_edge(u - 1, v - 1)  # Again convert to 0-index

        return G
