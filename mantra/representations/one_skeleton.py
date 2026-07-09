import networkx as nx
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx

from mantra.representations.internal import SimplexTrie


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
            Adjusted data object with all keys maintained and an
            `edge_index` tensor for representing the 1-skeleton being
            present.
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

        zero_simplices = sorted(
            node.simplex for node in simplex_trie.skeleton(0)
        )
        one_simplices = sorted(
            node.simplex for node in simplex_trie.skeleton(1)
        )

        G = nx.Graph()

        # Add every 0-simplex as a node *first*, in sorted (ascending) order.
        # `from_networkx` later maps nodes to consecutive integers in their
        # insertion order, so inserting them sorted makes that mapping the
        # identity. This keeps the node labels aligned with the original
        # (zero-indexed) vertex order, which other transforms -- e.g. the
        # moment-curve embedding -- rely on. Adding nodes lazily while
        # iterating over edges would instead permute the labels.
        for s in zero_simplices:
            (u,) = list(s)
            G.add_node(u - 1)  # Convert to 0-index

        # We loop over each pair of 0-simplices (u,v), where s = (u,v)  is a 1-simplex
        for s in one_simplices:
            # These are the 0-simplices composing a 1-simplex
            u, v = list(s)
            G.add_edge(u - 1, v - 1)  # Again convert to 0-index

        return G
