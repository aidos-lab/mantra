"""Deduplication utilities for manifold triangulations.

Detects duplicate triangulations that are combinatorially isomorphic
(same simplicial complex up to vertex relabeling). Uses a multi-level
filtering strategy for efficiency:

1. Cheap invariants (f-vector, degree sequence) to group triangulations
2. WL graph hash of the incidence graph to create subgroups
3. Full simplicial complex isomorphism checking within subgroups

The key insight is using the **incidence graph** (bipartite graph between
vertex nodes and top-simplex nodes) rather than the 1-skeleton. Many
3-manifold triangulations have the complete graph K_n as 1-skeleton,
making 1-skeleton-based isomorphism extremely slow (n! automorphisms).
The incidence graph encodes the full simplicial complex structure and
has far less symmetry.

Usage as a script::

    python -m mantra.deduplication path/to/manifolds.json
"""

import json
import sys
from collections import defaultdict
from itertools import combinations

import networkx as nx


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


def find_duplicates(triangulations, verbose=False):
    """Find duplicate triangulations in a dataset.

    Uses a three-level filtering strategy:
    1. Group by cheap invariants (f-vector + degree sequence)
    2. Subgroup by WL hash of the incidence graph
    3. Pairwise simplicial complex isomorphism within subgroups

    Parameters
    ----------
    triangulations : list of dict
        Each dict must have ``'triangulation'`` and ``'id'`` keys.
    verbose : bool
        If True, print progress to stderr.

    Returns
    -------
    list of tuple
        List of ``(id1, id2)`` pairs that are isomorphic duplicates.
    """
    n = len(triangulations)

    # Level 1: Group by cheap invariants
    if verbose:
        print(
            f"Computing invariants for {n} triangulations...",
            file=sys.stderr,
        )

    invariant_groups = defaultdict(list)
    for i, tri_data in enumerate(triangulations):
        tri = tri_data["triangulation"]
        key = compute_invariant_key(tri)
        invariant_groups[key].append(tri_data)
        if verbose and (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{n}", file=sys.stderr)

    nontrivial = {k: v for k, v in invariant_groups.items() if len(v) > 1}
    if verbose:
        print(
            f"  {len(invariant_groups)} invariant groups, "
            f"{len(nontrivial)} with >1 member",
            file=sys.stderr,
        )

    # Level 2 & 3: Within each group, subgroup by WL hash, then check
    duplicates = []
    total_groups = len(nontrivial)

    for group_idx, (key, members) in enumerate(nontrivial.items(), 1):
        if verbose:
            print(
                f"Group {group_idx}/{total_groups}: "
                f"f={key[0]}, deg_seq_len={len(key[1])}, "
                f"size={len(members)}",
                file=sys.stderr,
            )

        # Subgroup by WL hash of incidence graph
        wl_groups = defaultdict(list)
        for tri_data in members:
            wl = compute_wl_hash(tri_data["triangulation"])
            wl_groups[wl].append(tri_data)

        if verbose:
            nontrivial_wl = sum(1 for v in wl_groups.values() if len(v) > 1)
            max_wl = max(len(v) for v in wl_groups.values())
            print(
                f"  -> {len(wl_groups)} WL subgroups, "
                f"{nontrivial_wl} nontrivial, max size={max_wl}",
                file=sys.stderr,
            )

        # Pairwise isomorphism check within WL subgroups
        for wl_members in wl_groups.values():
            if len(wl_members) < 2:
                continue

            for i in range(len(wl_members)):
                for j in range(i + 1, len(wl_members)):
                    if are_isomorphic(
                        wl_members[i]["triangulation"],
                        wl_members[j]["triangulation"],
                    ):
                        duplicates.append(
                            (wl_members[i]["id"], wl_members[j]["id"])
                        )

    return duplicates


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python -m mantra.deduplication <path_to_json>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading {path}...", file=sys.stderr)

    with open(path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} triangulations.", file=sys.stderr)

    duplicates = find_duplicates(data, verbose=True)

    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate pairs:", file=sys.stderr)
        for id1, id2 in duplicates:
            print(f"  {id1} <-> {id2}")
        sys.exit(1)
    else:
        print("\nNo duplicates found.", file=sys.stderr)
        sys.exit(0)
