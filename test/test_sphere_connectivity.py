"""Subdivision variants + graph connectivity / moment-curve correctness.

Follow-up to ``test_sphere_pipeline.py``. Same two minimal spheres
(S^2 = boundary of a tetrahedron, S^3 = boundary of a 4-simplex), now
focused on:

1. **Stellar, partial stellar, and graded** subdivision -- each must keep
   the sphere a valid closed manifold (Euler characteristic + link
   condition) and hit the expected, deterministic simplex counts.
2. **Graph connectivity** of every representation the factory can build:
   - 1-skeleton  -> complete graph K_{d+2} on the minimal sphere,
   - dual graph  -> complete graph (every facet pair is adjacent),
   - Hasse/Levi  -> the bipartite-by-rank face-incidence graph.
3. **Moment-curve embedding** correctness: the curve values themselves,
   the barycentric propagation to higher simplices, and -- the question
   that motivated this -- that the per-node feature rows stay aligned
   with the graph the edges live on. (The embedding never *creates*
   ``edge_index``; the representation does, so we check both line up.)

The Levi (face-incidence) graph is printed explicitly. Run with ``-s``::

    pytest test/test_sphere_connectivity.py -s
"""

import random
from itertools import combinations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from mantra.augmentations.triangulation_2d import Triangulation2D
from mantra.augmentations.triangulation_3d import Triangulation3D
from mantra.representations.dual_graph import DualGraph
from mantra.representations.hasse_diagram import HasseDiagram
from mantra.representations.one_skeleton import OneSkeleton
from mantra.subdivision import (
    barycentric_stellar_graded,
    stellar_subdivision_raw,
)
from mantra.transforms.moment_curve_embedding import (
    MomentCurveEmbedding,
    _calculate_moment_curve,
)
from mantra.transforms.select_features import SelectFeatures
from mantra.transforms.structural_transforms import SetNumNodesTransform


S2_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
S3_TRI = [
    [1, 2, 3, 4],
    [1, 2, 3, 5],
    [1, 2, 4, 5],
    [1, 3, 4, 5],
    [2, 3, 4, 5],
]

# (name, dim, triangulation, n_vertices_0, n_top_0)
SPHERES = [
    ("S^2", 2, S2_TRI, 4, 4),
    ("S^3", 3, S3_TRI, 5, 5),
]


def _helper(dim, tri):
    cls = Triangulation2D if dim == 2 else Triangulation3D
    return cls([list(s) for s in tri])


def _undirected_edge_set(edge_index):
    return {
        frozenset((int(a), int(b)))
        for a, b in edge_index.t().tolist()
        if int(a) != int(b)
    }


# --- 1a. Full stellar subdivision --------------------------------------------


@pytest.mark.parametrize("name, dim, tri, nv0, ntop0", SPHERES)
def test_stellar_full(name, dim, tri, nv0, ntop0):
    # Full stellar: every top simplex gets a barycenter -> +1 vertex and the
    # facet is replaced by (d+1) new ones (net +d facets) per simplex.
    sub, n_v = stellar_subdivision_raw(tri, fraction=1.0)
    t = _helper(dim, sub)

    exp_nv = nv0 + ntop0
    exp_ntop = ntop0 * (dim + 1)
    print(
        f"\n[stellar full] {name}: nv {nv0}->{n_v}, n_top {ntop0}->{len(sub)}, "
        f"f_vector={t.f_vector()}, euler={t.euler_characteristic()}"
    )
    assert n_v == exp_nv
    assert len(sub) == exp_ntop
    assert t.euler_characteristic() == _helper(dim, tri).euler_characteristic()
    t.validate()

    # Stellar only adds barycenters, so the original labels survive verbatim.
    assert {v for s in tri for v in s} <= {v for s in sub for v in s}


# --- 1b. Partial stellar subdivision -----------------------------------------


@pytest.mark.parametrize("name, dim, tri, nv0, ntop0", SPHERES)
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_stellar_partial(name, dim, tri, nv0, ntop0, fraction):
    # Subdividing only a subset still leaves a valid complex because each
    # skipped facet keeps its boundary, so shared (d-1)-faces still match.
    sub, n_v = stellar_subdivision_raw(
        tri, fraction=fraction, rng=random.Random(7)
    )
    t = _helper(dim, sub)

    n_sub = round(fraction * ntop0)  # number of facets actually subdivided
    exp_nv = nv0 + n_sub
    exp_ntop = ntop0 + dim * n_sub
    print(
        f"[stellar {fraction:>4}] {name}: nv->{n_v} (exp {exp_nv}), "
        f"n_top->{len(sub)} (exp {exp_ntop}), euler={t.euler_characteristic()}"
    )
    assert n_v == exp_nv
    assert len(sub) == exp_ntop
    assert t.euler_characteristic() == _helper(dim, tri).euler_characteristic()
    t.validate()


def test_stellar_partial_is_seeded_deterministic():
    a = stellar_subdivision_raw(S3_TRI, fraction=0.5, rng=random.Random(0))
    b = stellar_subdivision_raw(S3_TRI, fraction=0.5, rng=random.Random(0))
    assert a == b


# --- 1c. Graded (vertex-targeted) subdivision --------------------------------


@pytest.mark.parametrize("name, dim, tri, nv0, ntop0", SPHERES)
@pytest.mark.parametrize("target", [6, 8, 10])
def test_graded_subdivision(name, dim, tri, nv0, ntop0, target):
    # Graded = repeated single-simplex stellar moves until the vertex target
    # is reached. Each move adds exactly 1 vertex and net d facets, so the
    # counts are determined by the target alone (independent of the seed).
    sub, n_v = barycentric_stellar_graded(
        tri, over_vrtx_cnt=target, rng=random.Random(3)
    )
    t = _helper(dim, sub)

    n_moves = target - nv0
    exp_ntop = ntop0 + dim * n_moves
    print(
        f"[graded n={target:>2}] {name}: nv->{n_v}, n_top->{len(sub)} "
        f"(exp {exp_ntop}), f_vector={t.f_vector()}, "
        f"euler={t.euler_characteristic()}"
    )
    assert n_v == target
    assert len(sub) == exp_ntop
    assert t.euler_characteristic() == _helper(dim, tri).euler_characteristic()
    t.validate()


# --- 2a. 1-skeleton connectivity ---------------------------------------------


@pytest.mark.parametrize("name, dim, tri, nv0, ntop0", SPHERES)
def test_one_skeleton_is_complete_graph(name, dim, tri, nv0, ntop0):
    # The 1-skeleton of the boundary of a simplex is the complete graph on
    # its vertices: every vertex pair spans an edge.
    out = OneSkeleton()(Data(triangulation=tri))
    edges = _undirected_edge_set(out.edge_index)
    all_pairs = {frozenset(p) for p in combinations(range(nv0), 2)}

    print(
        f"\n[1-skeleton] {name}: nodes={nv0}, undirected_edges={len(edges)} "
        f"(complete graph needs {len(all_pairs)})"
    )
    assert edges == all_pairs  # K_{nv0}
    # Undirected (each edge appears in both directions) and no self-loops.
    assert out.edge_index.shape[1] == 2 * len(all_pairs)
    assert all(int(a) != int(b) for a, b in out.edge_index.t().tolist())
    assert int(out.edge_index.max()) < nv0


# --- 2b. Dual-graph connectivity ---------------------------------------------


@pytest.mark.parametrize("name, dim, tri, nv0, ntop0", SPHERES)
def test_dual_graph_connectivity(name, dim, tri, nv0, ntop0):
    # On the boundary of a simplex every pair of facets shares a
    # codimension-1 face, so the dual graph is complete on the facets.
    out = DualGraph()(Data(triangulation=tri, dimension=dim))
    edges = _undirected_edge_set(out.edge_index)
    all_pairs = {frozenset(p) for p in combinations(range(ntop0), 2)}

    print(
        f"[dual graph] {name}: facet-nodes={ntop0}, "
        f"undirected_edges={len(edges)} (complete needs {len(all_pairs)})"
    )
    assert edges == all_pairs
    assert int(out.edge_index.max()) < ntop0


# --- 2c. Hasse / Levi (face-incidence) graph ---------------------------------


def _build_levi(dim, tri):
    """Return the nx Hasse/Levi graph with simplex-tuple node labels."""
    tops = sorted({tuple(s) for s in tri})
    tops.sort(key=len)
    return HasseDiagram()._build_hasse_diagram(tops, {})


@pytest.mark.parametrize(
    "name, dim, tri, per_rank, n_edges",
    [
        ("S^2", 2, S2_TRI, {0: 4, 1: 6, 2: 4}, 24),
        ("S^3", 3, S3_TRI, {0: 5, 1: 10, 2: 10, 3: 5}, 70),
    ],
)
def test_levi_graph_structure(name, dim, tri, per_rank, n_edges):
    G = _build_levi(dim, tri)

    by_rank = {}
    for node in G.nodes():
        by_rank.setdefault(len(node) - 1, []).append(node)
    counts = {r: len(v) for r, v in by_rank.items()}

    assert counts == per_rank
    assert G.number_of_edges() == n_edges

    # Every edge joins consecutive ranks only (covering relation) and the
    # lower simplex is a strict face of the higher one -> a genuine
    # face-incidence (Levi) graph, bipartite between each adjacent rank pair.
    for u, v in G.edges():
        lo, hi = sorted((u, v), key=len)
        assert len(hi) - len(lo) == 1
        assert set(lo) < set(hi)


def test_print_levi_graph_s2():
    """Explicitly print the Levi (face-incidence) graph of S^2."""
    G = _build_levi(2, S2_TRI)
    rank_name = {0: "vertices", 1: "edges", 2: "triangles"}

    print("\n########## Levi / Hasse face-incidence graph of S^2 ##########")
    by_rank = {}
    for node in G.nodes():
        by_rank.setdefault(len(node) - 1, []).append(tuple(sorted(node)))
    for r in sorted(by_rank):
        simplices = sorted(by_rank[r])
        print(f"  rank {r} ({rank_name[r]}, {len(simplices)}): {simplices}")

    print("  incidence edges (face  <  coface):")
    for u, v in sorted(
        ((tuple(sorted(a)), tuple(sorted(b))) for a, b in G.edges()),
        key=lambda e: (len(e[0]) + len(e[1]), e),
    ):
        lo, hi = sorted((u, v), key=len)
        print(f"    {lo}  <  {hi}")

    # 12 vertex-edge incidences + 12 edge-triangle incidences = 24.
    assert G.number_of_edges() == 24


# --- 3. Moment-curve embedding correctness + alignment -----------------------


@pytest.mark.parametrize("name, dim, tri, nv0, ntop0", SPHERES)
def test_moment_curve_values_and_propagation(name, dim, tri, nv0, ntop0):
    # The bare embedding (propagate=True) must (a) reproduce the analytic
    # moment curve at the vertices and (b) place every higher simplex at the
    # barycenter of its vertices' curve points.
    data = Data(triangulation=tri, n_vertices=nv0, dimension=dim)
    out = MomentCurveEmbedding(propagate=True)(data)
    vals = out.moment_curve_embedding

    X = _calculate_moment_curve(nv0, dim)
    assert X.shape == (nv0, 2 * dim + 1)
    assert np.allclose(vals[0].numpy(), X)

    # Check the barycenter property on the lexicographically-first simplex of
    # each rank by reconstructing the simplex ordering the helper uses.
    print(f"\n[moment curve] {name}: vertex coords shape {tuple(vals[0].shape)}")
    simplices = set()
    for s in tri:
        for r in range(1, dim + 1):
            simplices.update(combinations(s, r + 1))
    for r in range(1, dim + 1):
        ordered = sorted([s for s in simplices if len(s) == r + 1])
        first = ordered[0]
        expected = X[np.array(first) - 1].mean(axis=0)
        assert np.allclose(vals[r][0].numpy(), expected), (r, first)
        print(f"   rank {r}: {len(ordered)} simplices, "
              f"barycenter[{first}] = {np.round(expected, 3).tolist()}")


def test_moment_curve_aligned_with_one_skeleton():
    # On the 1-skeleton the graph nodes ARE the vertices, so x[i] must equal
    # the moment-curve point of vertex i, and edge_index must index into the
    # same rows. This is the "do the edges and features line up?" check.
    dim, tri, nv0 = 2, S2_TRI, 4
    data = Data(triangulation=tri, n_vertices=nv0, dimension=dim)
    data = OneSkeleton()(data)
    data = SetNumNodesTransform()(data)
    data = MomentCurveEmbedding(propagate=False)(data)
    data = SelectFeatures(
        src="moment_curve_embedding", dst=None, representation="graph"
    )(data)

    X = _calculate_moment_curve(nv0, dim)
    assert data.x.shape == (nv0, 2 * dim + 1)
    assert np.allclose(data.x.numpy(), X)  # row i == vertex i
    assert int(data.num_nodes) == nv0
    # edge_index references valid rows of x (no dangling node).
    assert int(data.edge_index.max()) < data.x.shape[0]
    print(
        f"\n[align] one_skeleton+mc S^2: x{tuple(data.x.shape)} rows == "
        f"vertices, edge_index max {int(data.edge_index.max())} < "
        f"{data.x.shape[0]} nodes -> aligned"
    )


@pytest.mark.parametrize(
    "rep_name, rep, n_expected_nodes_s2",
    [
        ("one_skeleton", OneSkeleton, 4),  # nodes = vertices
        ("dual", DualGraph, 4),  # nodes = facets (triangles)
        ("hasse", HasseDiagram, 14),  # nodes = all simplices
    ],
)
def test_feature_row_count_matches_node_count(
    rep_name, rep, n_expected_nodes_s2
):
    # For every graph representation, after SetNumNodes the moment-curve
    # feature matrix has exactly one row per graph node and edge_index never
    # references a row that doesn't exist -- so features and connectivity are
    # consistent regardless of which representation builds the graph.
    data = Data(triangulation=S2_TRI, n_vertices=4, dimension=2)
    data = rep()(data)
    data = SetNumNodesTransform()(data)
    data = MomentCurveEmbedding(propagate=False)(data)
    data = SelectFeatures(
        src="moment_curve_embedding", dst=None, representation="graph"
    )(data)

    assert int(data.num_nodes) == n_expected_nodes_s2
    assert data.x.shape[0] == n_expected_nodes_s2
    if "edge_index" in data and data.edge_index.numel() > 0:
        assert int(data.edge_index.max()) < data.x.shape[0]
    print(
        f"[row-count] {rep_name:>13}: num_nodes={int(data.num_nodes)}, "
        f"x rows={data.x.shape[0]} -> match"
    )
