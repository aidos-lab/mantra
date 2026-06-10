"""End-to-end pipeline + subdivision walk-through for a single sphere.

This module takes *one* minimal sphere triangulation in each dimension
(S^2 = boundary of a tetrahedron, S^3 = boundary of a 4-simplex) and:

1. runs it step-by-step through the graph pre-transform pipeline that the
   factory assembles, asserting and printing the output of every stage;
2. runs it end-to-end through :func:`mantra.datasets.factory.make_dataset`
   for every legal (representation, featurization) combination and for the
   ``cwn`` model bundle, loading the processed dataset from disk;
3. barycentrically subdivides it and pushes the *subdivided* sphere back
   through the factory (``subdivision_level=1``), confirming the refined
   manifold is still a valid sphere.

The factory and barycentric subdivision are the focus.

Every check also prints a human-readable summary. Run with ``-s`` to read
the printed report and eyeball the numbers manually::

    pytest test/test_sphere_pipeline.py -s
"""

import json
import os

import pytest
import torch

from mantra.augmentations.triangulation_2d import Triangulation2D
from mantra.augmentations.triangulation_3d import Triangulation3D
from mantra.datasets.factory import make_dataset
from mantra.subdivision import (
    barycentric_subdivision_raw,
    subdivide_dataset_json,
)
from mantra.transforms.attribute_transform import (
    NodeDegreeTransform,
    NodeRandomTransform,
)
from mantra.transforms.select_features import SelectFeatures
from mantra.transforms.structural_transforms import SetNumNodesTransform
from mantra.transforms.task_transforms import (
    NAME_TO_CLASS_2M,
    NAME_TO_CLASS_3M,
    NameToClass2MTransform,
    NameToClass3MTransform,
)
from mantra.representations.one_skeleton import OneSkeleton
from torch_geometric.data import Data


# --- Minimal sphere triangulations (full dataset-style entries) --------------

# Boundary of a tetrahedron == minimal triangulation of S^2.
S2_ENTRY = {
    "id": "s2-minimal",
    "triangulation": [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]],
    "dimension": 2,
    "n_vertices": 4,
    "betti_numbers": [1, 0, 1],  # S^2: b0=1, b1=0, b2=1 (orientable)
    "name": "S^2",
    "genus": 0,
    "orientable": True,
}

# Boundary of a 4-simplex == minimal triangulation of S^3.
S3_ENTRY = {
    "id": "s3-minimal",
    "triangulation": [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 4, 5],
        [1, 3, 4, 5],
        [2, 3, 4, 5],
    ],
    "dimension": 3,
    "n_vertices": 5,
    "betti_numbers": [1, 0, 0, 1],  # S^3
    "name": "S^3",
    "genus": 0,
    "orientable": True,
}

SPHERES = {2: S2_ENTRY, 3: S3_ENTRY}


def _entry_as_data(entry):
    return Data(**entry)


def _triangulation_helper(dim, top_simplices):
    cls = Triangulation2D if dim == 2 else Triangulation3D
    return cls([list(s) for s in top_simplices])


# --- 1. Step-by-step trace through the graph pipeline ------------------------


@pytest.mark.parametrize("dim", [2, 3])
def test_pipeline_steps_traced(dim):
    """Apply each factory pipeline stage in order, printing the output.

    Mirrors the ``one_skeleton`` + ``random`` + ``name`` recipe that
    ``make_dataset`` composes, but applies the stages one at a time so the
    incremental output is visible and individually asserted.
    """
    torch.manual_seed(0)  # make the random features reproducible for the print
    entry = SPHERES[dim]
    name_cls = NAME_TO_CLASS_2M if dim == 2 else NAME_TO_CLASS_3M
    task_tf = NameToClass2MTransform() if dim == 2 else NameToClass3MTransform()
    feature_dim = 8

    data = _entry_as_data(entry)
    print(f"\n########## {entry['name']} (dim={dim}) pipeline trace ##########")
    print(f"[0] input keys            : {sorted(data.keys())}")
    print(f"    triangulation         : {entry['triangulation']}")

    # Stage 1: representation (1-skeleton graph).
    data = OneSkeleton()(data)
    assert "edge_index" in data
    n_edges_directed = data.edge_index.shape[1]
    print(f"[1] OneSkeleton           : edge_index {tuple(data.edge_index.shape)}"
          f" ({n_edges_directed // 2} undirected edges)")

    # Stage 2: num_nodes from n_vertices.
    data = SetNumNodesTransform()(data)
    assert int(data.num_nodes) == entry["n_vertices"]
    print(f"[2] SetNumNodesTransform  : num_nodes = {int(data.num_nodes)}")

    # Stage 3: random node featurization.
    data = NodeRandomTransform(dim=feature_dim, propagate=False)(data)
    assert data.random_features.shape == (entry["n_vertices"], feature_dim)
    print(f"[3] NodeRandomTransform   : random_features "
          f"{tuple(data.random_features.shape)}")

    # Stage 4: promote the chosen feature to the canonical ``x``.
    data = SelectFeatures(src="random_features", dst=None, representation="graph")(
        data
    )
    assert data.x.shape == (entry["n_vertices"], feature_dim)
    print(f"[4] SelectFeatures        : x {tuple(data.x.shape)}")

    # Stage 5: classification label in canonical space.
    data = task_tf(data)
    assert data.y.tolist() == [name_cls[entry["name"]]]
    assert data.y.dtype == torch.long
    print(f"[5] NameToClass{dim}M       : y = {data.y.tolist()} "
          f"(canonical class for {entry['name']!r})")
    print(f"    final keys            : {sorted(data.keys())}")


# --- 2. Factory end-to-end over every legal (representation, featurization) ---

# (representation, featurization) pairs that are legal on both dims.
_GRAPH_RECIPES = [
    ("one_skeleton", "random"),
    ("one_skeleton", "node_degree"),
    ("one_skeleton", "mc"),
    ("dual", "node_degree"),
    ("hasse", "mc"),
]


@pytest.fixture
def sphere_root(tmp_path):
    """Per-test on-disk dataset root with both sphere JSONs written out."""
    root = str(tmp_path)
    paths = {}
    for dim, entry in SPHERES.items():
        p = os.path.join(root, f"sphere_{dim}d.json")
        with open(p, "w") as f:
            json.dump([entry], f)
        paths[dim] = p
    return root, paths


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("representation, featurization", _GRAPH_RECIPES)
def test_factory_graph_paths(dim, representation, featurization, sphere_root):
    root, paths = sphere_root
    entry = SPHERES[dim]
    name_cls = NAME_TO_CLASS_2M if dim == 2 else NAME_TO_CLASS_3M
    feature_dim = 8

    ds = make_dataset(
        root=root,
        dimension=dim,
        representation=representation,
        featurization=featurization,
        task="name",
        balanced=False,
        local_path=paths[dim],
        feature_dim=feature_dim,
        force_reload=True,
        name=f"trace_{dim}d_{representation}_{featurization}",
    )
    assert len(ds) == 1
    d = ds[0]

    # The label is always assigned in canonical class space.
    assert d.y.tolist() == [name_cls[entry["name"]]]

    # Featurization always lands a feature matrix on ``data.x``.
    assert "x" in d, f"no x produced for {representation}/{featurization}"
    assert d.x.shape[0] > 0

    print(
        f"\n[factory dim={dim}] {representation:>13} + {featurization:<11} -> "
        f"x={tuple(d.x.shape)}, y={d.y.tolist()}, keys={sorted(d.keys())}"
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_factory_bundle_cwn(dim, sphere_root):
    """The higher-order ``cwn`` bundle emits per-rank features x_0..x_d."""
    root, paths = sphere_root
    entry = SPHERES[dim]
    name_cls = NAME_TO_CLASS_2M if dim == 2 else NAME_TO_CLASS_3M

    ds = make_dataset(
        root=root,
        dimension=dim,
        bundle="cwn",
        featurization="random",
        task="name",
        balanced=False,
        local_path=paths[dim],
        feature_dim=4,
        force_reload=True,
        name=f"trace_{dim}d_bundle_cwn",
    )
    d = ds[0]

    assert d.y.tolist() == [name_cls[entry["name"]]]
    # One feature block per rank 0..dim.
    for r in range(dim + 1):
        assert f"x_{r}" in d, f"missing x_{r} for cwn bundle (dim={dim})"
        assert d[f"x_{r}"].shape[1] == 4
    # Incidence matrices the bundle keys its propagation off of.
    assert "incidence_0" in d

    ranks = {f"x_{r}": tuple(d[f"x_{r}"].shape) for r in range(dim + 1)}
    print(f"\n[factory dim={dim}] bundle=cwn -> y={d.y.tolist()}, per-rank x: {ranks}")


# --- 3. Barycentric subdivision of the sphere, then back through the factory --


@pytest.mark.parametrize(
    "dim, exp_nv, exp_ntop, exp_fvec",
    [
        (2, 14, 24, (14, 36, 24)),
        (3, 30, 120, (30, 150, 240, 120)),
    ],
)
def test_barycentric_subdivision_then_factory(
    dim, exp_nv, exp_ntop, exp_fvec, tmp_path
):
    entry = SPHERES[dim]

    # (a) Subdivide the raw triangulation and check it is the same manifold.
    sub_tri, n_v = barycentric_subdivision_raw(entry["triangulation"])
    t = _triangulation_helper(dim, sub_tri)
    orig = _triangulation_helper(dim, entry["triangulation"])

    print(f"\n########## {entry['name']} barycentric subdivision ##########")
    print(f"  original : n_v={entry['n_vertices']}, n_top={len(entry['triangulation'])}, "
          f"f_vector={orig.f_vector()}, euler={orig.euler_characteristic()}")
    print(f"  subdivided: n_v={n_v}, n_top={len(sub_tri)}, "
          f"f_vector={t.f_vector()}, euler={t.euler_characteristic()}")

    assert n_v == exp_nv
    assert len(sub_tri) == exp_ntop
    assert t.f_vector() == exp_fvec
    # Same homeomorphism type -> identical Euler characteristic + valid manifold.
    assert t.euler_characteristic() == orig.euler_characteristic()
    t.validate()

    # (b) Feed the subdivided sphere through the factory at subdivision_level=1.
    sub_entry = subdivide_dataset_json([entry], n_subdivisions=1)
    assert sub_entry[0]["n_vertices"] == exp_nv
    sub_path = os.path.join(str(tmp_path), f"sphere_{dim}d_bary1.json")
    with open(sub_path, "w") as f:
        json.dump(sub_entry, f)

    ds = make_dataset(
        root=str(tmp_path),
        dimension=dim,
        representation="one_skeleton",
        featurization="random",
        task="name",
        balanced=False,
        local_path=sub_path,
        feature_dim=8,
        force_reload=True,
        subdivision_level=1,
        name=f"trace_{dim}d_bary1",
    )
    d = ds[0]
    name_cls = NAME_TO_CLASS_2M if dim == 2 else NAME_TO_CLASS_3M

    assert int(d.num_nodes) == exp_nv
    assert d.x.shape == (exp_nv, 8)
    assert d.y.tolist() == [name_cls[entry["name"]]]
    print(
        f"  factory(subdivision_level=1): num_nodes={int(d.num_nodes)}, "
        f"x={tuple(d.x.shape)}, edge_index={tuple(d.edge_index.shape)}, "
        f"y={d.y.tolist()}"
    )
