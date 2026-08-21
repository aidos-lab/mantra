"""Utility functions for interacting with triangulations."""

import json
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from itertools import combinations

import networkx as nx


def store_triangulations(triangulations, output=None):
    """Store triangulations in "pretty" format.

    This function stores a list of triangulations in a somewhat
    "prettified" JSON format (one line per simplex). The output
    may either be `stdout` or a file.

    Parameters
    ----------
    triangulations : list of dict
        List of triangulations to store.

    output : str or None
        Output file. If `None`, will write the list to `stdout`.
    """
    with (
        open(output, "w") if output is not None else nullcontext(sys.stdout)
    ) as f:
        result = json.dumps(triangulations, indent=2)

        regex = re.compile(
            r"^(\s+)\[(.*?)\]([,]\s+?)", re.MULTILINE | re.DOTALL
        )

        def prettify_triangulation(match):
            """Auxiliary function for pretty-printing a triangulation.

            Given a match that contains *all* the top-level vertices
            involved in the triangulation, this function will ensure
            that they are all printed on individual lines. Plus, any
            indent is preserved.
            """
            groups = match.groups()
            indent = match.group(1)
            vertex = match.group(2)
            vertex = vertex.replace("\n", "")
            vertex = re.sub(r"\s+", "", vertex)

            result = f"{indent}[{vertex}]"

            if len(groups) == 3:
                result += ",\n"

            return result

        result = regex.sub(prettify_triangulation, result)

        # Fix indent of "triangulation" fields afterwards. This ensures
        # that the closing bracket of the triangulation key aligns with
        # the start.
        regex = re.compile(
            r"^(\s+)\"triangulation\":.*?\]\]", re.MULTILINE | re.DOTALL
        )

        indents = [len(match.group(1)) for match in regex.finditer(result)]

        assert len(indents) != 0
        assert indents[0] > 0
        assert sum(indents) / indents[0] == len(indents)

        indent = " " * indents[0]
        result = result.replace("]],", f"]\n{indent}],")

        f.write(result)


def compute_f_vector(triangulation):
    """Compute the f-vector of a triangulation.

    Parameters
    ----------
    triangulation : list of list of int
        Top-level simplices of the triangulation.

    Returns
    -------
    tuple of int
        (n_vertices, n_edges, n_faces, ..., n_top_simplices)
    """
    dim = len(triangulation[0]) - 1
    faces = [set() for _ in range(dim + 1)]

    for simplex in triangulation:
        simplex_sorted = tuple(sorted(simplex))
        faces[dim].add(simplex_sorted)
        for d in range(dim):
            for face in combinations(simplex_sorted, d + 1):
                faces[d].add(face)

    return tuple(len(f) for f in faces)


def compute_degree_sequence(triangulation):
    """Compute sorted vertex degree sequence of the 1-skeleton.

    Parameters
    ----------
    triangulation : list of list of int
        Top-level simplices of the triangulation.

    Returns
    -------
    tuple of int
        Sorted degree sequence.
    """
    degree = defaultdict(int)
    edges = set()
    for simplex in triangulation:
        for u, v in combinations(simplex, 2):
            edge = (min(u, v), max(u, v))
            if edge not in edges:
                edges.add(edge)
                degree[u] += 1
                degree[v] += 1

    return tuple(sorted(degree.values()))


def _build_incidence_graph(triangulation):
    """Build the incidence graph of a simplicial complex.

    The incidence graph is a bipartite graph with two types of nodes:
    - Vertex nodes (labeled ``('v', vertex_id)``) with ``node_type='v'``
    - Top-simplex nodes (labeled ``('t', index)``) with ``node_type='t'``

    An edge connects a vertex node to a top-simplex node iff the vertex
    belongs to that simplex. This graph encodes the full combinatorial
    structure of the simplicial complex and is much more discriminating
    than the 1-skeleton (which can be K_n for many 3-manifold
    triangulations, yielding n! automorphisms).
    """
    G = nx.Graph()
    vertices = sorted(set(v for simplex in triangulation for v in simplex))
    for v in vertices:
        G.add_node(("v", v), node_type="v")

    for i, simplex in enumerate(triangulation):
        G.add_node(("t", i), node_type="t")
        for v in simplex:
            G.add_edge(("v", v), ("t", i))

    return G


def compute_edge_simplex_count_sequence(triangulation):
    """Compute sorted sequence of per-edge top-simplex counts.

    For each edge in the triangulation, counts how many top-level
    simplices contain that edge. This is a powerful invariant that
    discriminates well even when the 1-skeleton is the complete graph.

    Parameters
    ----------
    triangulation : list of list of int
        Top-level simplices of the triangulation.

    Returns
    -------
    tuple of int
        Sorted sequence of edge-simplex counts.
    """
    edge_count = defaultdict(int)
    for simplex in triangulation:
        for u, v in combinations(simplex, 2):
            edge_count[(min(u, v), max(u, v))] += 1

    return tuple(sorted(edge_count.values()))


def compute_invariant_key(triangulation):
    """Compute a hashable invariant key for cheap grouping.

    Combines the f-vector, sorted degree sequence, and sorted
    edge-simplex-count sequence. The edge-simplex-count is
    particularly powerful for 3-manifold triangulations where
    the 1-skeleton is often the complete graph K_n.

    Parameters
    ----------
    triangulation : list of list of int
        Top-level simplices of the triangulation.

    Returns
    -------
    tuple
        Hashable key (f_vector, degree_sequence, edge_simplex_counts).
    """
    f_vec = compute_f_vector(triangulation)
    deg_seq = compute_degree_sequence(triangulation)
    edge_tet_seq = compute_edge_simplex_count_sequence(triangulation)
    return (f_vec, deg_seq, edge_tet_seq)


def compute_wl_hash(triangulation, iterations=5):
    """Compute Weisfeiler-Lehman graph hash of the incidence graph.

    Uses the incidence graph (bipartite between vertices and top-simplices)
    rather than the 1-skeleton, giving much better discrimination.

    Parameters
    ----------
    triangulation : list of list of int
        Top-level simplices of the triangulation.
    iterations : int
        Number of WL refinement iterations.

    Returns
    -------
    str
        Hexadecimal hash string.
    """
    G = _build_incidence_graph(triangulation)
    return nx.weisfeiler_lehman_graph_hash(
        G, node_attr="node_type", iterations=iterations
    )


def _node_type_match(n1_attrs, n2_attrs):
    """Node match function preserving the bipartition."""
    return n1_attrs["node_type"] == n2_attrs["node_type"]


def are_isomorphic(tri1, tri2):
    """Check if two triangulations are isomorphic as simplicial complexes.

    Uses isomorphism of the incidence graph (bipartite between vertex
    nodes and top-simplex nodes) with node-type matching. This correctly
    captures full simplicial complex isomorphism: an incidence graph
    isomorphism that maps vertex nodes to vertex nodes and simplex nodes
    to simplex nodes corresponds exactly to a vertex relabeling that
    maps one complex onto the other.

    Parameters
    ----------
    tri1, tri2 : list of list of int
        Top-level simplices of the two triangulations.

    Returns
    -------
    bool
        True if the triangulations are isomorphic.
    """
    G1 = _build_incidence_graph(tri1)
    G2 = _build_incidence_graph(tri2)

    GM = nx.algorithms.isomorphism.GraphMatcher(
        G1, G2, node_match=_node_type_match
    )
    return GM.is_isomorphic()
