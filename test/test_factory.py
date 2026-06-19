"""End-to-end tests for the dataset factory (``mantra.build_dataset``).

All datasets are built from a small local fixture (no network). Graph configs
are checked for the canonical ``edge_index``/``x``/``y`` payload; higher-order
bundles are checked against the exact key + shape contract that the SAN, SCCNN
and CWN models consume in the benchmarks repo.
"""

import json
import os

import pytest

from mantra import DatasetFactoryConfig, build_dataset
from mantra.factory import bundle_required_keys

# Boundary of a tetrahedron: a 4-vertex 2-sphere (a valid closed surface).
SPHERE_2D = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
# Boundary of a 4-simplex: a 5-vertex 3-sphere (a valid closed 3-manifold).
SPHERE_3D = [
    [1, 2, 3, 4],
    [1, 2, 3, 5],
    [1, 2, 4, 5],
    [1, 3, 4, 5],
    [2, 3, 4, 5],
]


def _entry(idx, name, orientable, triangulation, dimension):
    verts = sorted({v for s in triangulation for v in s})
    return {
        "id": f"m{idx}",
        "triangulation": [list(s) for s in triangulation],
        "dimension": dimension,
        "n_vertices": len(verts),
        "betti_numbers": [1, 0, 1],
        "torsion_coefficients": ["", "", ""],
        "name": name,
        "genus": 0,
        "orientable": orientable,
        "vertex_transitive": True,
    }


@pytest.fixture
def base_2d(tmp_path):
    """Two-class 2D base JSON (S^2 orientable, RP^2 non-orientable)."""
    entries = [_entry(i, "S^2", True, SPHERE_2D, 2) for i in range(3)] + [
        _entry(i + 3, "RP^2", False, SPHERE_2D, 2) for i in range(3)
    ]
    path = tmp_path / "base_2d.json"
    path.write_text(json.dumps(entries))
    return str(path)


@pytest.fixture
def base_3d(tmp_path):
    """Single-class 3D base JSON of 3-spheres."""
    entries = [_entry(i, "S^3", True, SPHERE_3D, 3) for i in range(3)]
    path = tmp_path / "base_3d.json"
    path.write_text(json.dumps(entries))
    return str(path)


def _cfg(tmp_path, base, sub="g", **kwargs):
    return DatasetFactoryConfig(
        root=str(tmp_path / sub), local_path=base, **kwargs
    )


def _dim(t):
    """Number of rows of a (sparse or dense) 2D tensor."""
    return t.shape[0]


# --------------------------------------------------------------------------- #
# Graph representation x featurization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "representation", ["one_skeleton", "dual_graph", "hasse_diagram"]
)
@pytest.mark.parametrize("featurization", ["random", "degree"])
def test_graph_configs_produce_node_features(
    base_2d, tmp_path, representation, featurization
):
    cfg = _cfg(
        tmp_path,
        base_2d,
        sub=f"{representation}-{featurization}",
        dimension=2,
        variant="unbalanced",
        task="name",
        representation=representation,
        featurization=featurization,
    )
    ds = build_dataset(cfg)
    data = ds[0]
    assert "edge_index" in data
    assert "x" in data
    # One feature row per graph node.
    n_nodes = int(data.edge_index.max().item()) + 1
    assert data.x.shape[0] == n_nodes
    assert "y" in data


def test_moment_curve_on_one_skeleton(base_2d, tmp_path):
    cfg = _cfg(
        tmp_path,
        base_2d,
        sub="mc",
        dimension=2,
        variant="unbalanced",
        task="name",
        representation="one_skeleton",
        featurization="moment_curve",
    )
    ds = build_dataset(cfg)
    # Moment-curve is a per-vertex embedding; one_skeleton keeps n_vertices.
    assert ds[0].x.shape[0] == int(ds[0].n_vertices)


# --------------------------------------------------------------------------- #
# Higher-order bundles
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bundle", ["san", "sccnn", "cwn"])
def test_bundle_required_keys_present(base_2d, tmp_path, bundle):
    cfg = _cfg(
        tmp_path,
        base_2d,
        sub=bundle,
        dimension=2,
        variant="unbalanced",
        task="orientability",
        bundle=bundle,
    )
    ds = build_dataset(cfg)
    keys = set(ds[0].keys())
    assert bundle_required_keys(bundle, 2).issubset(keys)


def test_sccnn_shape_contract(base_2d, tmp_path):
    cfg = _cfg(
        tmp_path,
        base_2d,
        sub="sccnn-shape",
        dimension=2,
        variant="unbalanced",
        task="name",
        bundle="sccnn",
    )
    data = build_dataset(cfg)[0]
    # Feature row counts chain through the incidence (boundary) matrices:
    # incidence_1: n_0 x n_1, incidence_2: n_1 x n_2.
    assert data.x_0.shape[0] == data.incidence_1.shape[0]
    assert data.x_1.shape[0] == data.incidence_1.shape[1]
    assert data.x_1.shape[0] == data.incidence_2.shape[0]
    assert data.x_2.shape[0] == data.incidence_2.shape[1]
    # Laplacians are square on their dimension.
    assert data.hodge_laplacian_0.shape == (_dim(data.x_0), _dim(data.x_0))
    assert data.up_laplacian_1.shape == (_dim(data.x_1), _dim(data.x_1))
    assert data.down_laplacian_1.shape == (_dim(data.x_1), _dim(data.x_1))


def test_cwn_shape_contract(base_2d, tmp_path):
    cfg = _cfg(
        tmp_path,
        base_2d,
        sub="cwn-shape",
        dimension=2,
        variant="unbalanced",
        task="name",
        bundle="cwn",
    )
    data = build_dataset(cfg)[0]
    assert data.adjacency_1.shape == (_dim(data.x_1), _dim(data.x_1))
    assert data.x_0.shape[0] == data.incidence_1.shape[0]
    assert data.x_2.shape[0] == data.incidence_2.shape[1]


def test_san_shape_contract(base_2d, tmp_path):
    cfg = _cfg(
        tmp_path,
        base_2d,
        sub="san-shape",
        dimension=2,
        variant="unbalanced",
        task="orientability",
        bundle="san",
    )
    data = build_dataset(cfg)[0]
    assert data.up_laplacian_1.shape == (_dim(data.x_1), _dim(data.x_1))
    assert data.down_laplacian_1.shape == (_dim(data.x_1), _dim(data.x_1))


def test_bundle_on_3d(base_3d, tmp_path):
    cfg = _cfg(
        tmp_path,
        base_3d,
        sub="sccnn-3d",
        dimension=3,
        variant="unbalanced",
        task="name",
        bundle="sccnn",
    )
    ds = build_dataset(cfg)
    keys = set(ds[0].keys())
    # 3D (sc_order > 2) additionally requires the up Laplacian on faces.
    assert bundle_required_keys("sccnn", 3).issubset(keys)
    assert "up_laplacian_2" in keys


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def test_orientability_labels(base_2d, tmp_path):
    cfg = _cfg(
        tmp_path,
        base_2d,
        sub="orient",
        dimension=2,
        variant="unbalanced",
        task="orientability",
        representation="one_skeleton",
        featurization="random",
    )
    ds = build_dataset(cfg)
    for data in ds:
        assert int(data.y.item()) == int(data.orientable)


def test_name_labels_distinct(base_2d, tmp_path):
    cfg = _cfg(
        tmp_path,
        base_2d,
        sub="name",
        dimension=2,
        variant="unbalanced",
        task="name",
        representation="dual_graph",
        featurization="random",
    )
    ds = build_dataset(cfg)
    assert set(ds.label_to_index) == {"S^2", "RP^2"}
    assert len(set(ds.label_to_index.values())) == 2


# --------------------------------------------------------------------------- #
# Subdivided variants through the factory
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "variant", ["barycentric", "stellar_full", "stellar_0.75", "graded"]
)
def test_subdivided_variants_build(base_2d, tmp_path, variant):
    cfg = _cfg(
        tmp_path,
        base_2d,
        sub=variant,
        dimension=2,
        variant=variant,
        task="name",
        bundle="cwn",
    )
    ds = build_dataset(cfg)
    assert len(ds) >= 1
    assert bundle_required_keys("cwn", 2).issubset(set(ds[0].keys()))


# --------------------------------------------------------------------------- #
# Caching / coexistence
# --------------------------------------------------------------------------- #


def test_distinct_configs_get_distinct_processed_dirs(base_2d, tmp_path):
    root = tmp_path / "shared"
    cfg_a = DatasetFactoryConfig(
        root=str(root),
        local_path=base_2d,
        dimension=2,
        variant="unbalanced",
        task="name",
        bundle="san",
    )
    cfg_b = DatasetFactoryConfig(
        root=str(root),
        local_path=base_2d,
        dimension=2,
        variant="unbalanced",
        task="name",
        bundle="cwn",
    )
    ds_a = build_dataset(cfg_a)
    ds_b = build_dataset(cfg_b)
    assert cfg_a.slug() in ds_a.processed_dir
    assert cfg_b.slug() in ds_b.processed_dir
    assert ds_a.processed_dir != ds_b.processed_dir


def test_cached_reload_preserves_label_to_index(base_2d, tmp_path):
    kwargs = dict(
        root=str(tmp_path / "cache"),
        local_path=base_2d,
        dimension=2,
        variant="unbalanced",
        task="name",
        representation="dual_graph",
        featurization="random",
    )
    first = build_dataset(DatasetFactoryConfig(**kwargs))
    processed = first.processed_paths[0]
    assert os.path.exists(processed)
    mtime = os.path.getmtime(processed)

    # Rebuilding loads from cache (no reprocessing) yet still exposes the map,
    # reconstructed from the stored label/y pairs.
    second = build_dataset(DatasetFactoryConfig(**kwargs))
    assert os.path.getmtime(processed) == mtime
    assert second.label_to_index == first.label_to_index
    assert set(second.label_to_index) == {"S^2", "RP^2"}


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #


def test_validation_errors(tmp_path):
    common = dict(root=str(tmp_path), dimension=2, variant="unbalanced")
    with pytest.raises(ValueError):  # bundle + representation
        DatasetFactoryConfig(
            bundle="san", representation="dual_graph", **common
        )
    with pytest.raises(ValueError):  # neither family
        DatasetFactoryConfig(**common)
    with pytest.raises(ValueError):  # unknown variant
        DatasetFactoryConfig(
            root=str(tmp_path), dimension=2, variant="nope", bundle="san"
        )
    with pytest.raises(ValueError):  # bad dimension
        DatasetFactoryConfig(
            root=str(tmp_path), dimension=5, variant="unbalanced", bundle="san"
        )
    with pytest.raises(ValueError):  # unknown task
        DatasetFactoryConfig(task="nope", bundle="san", **common)
    with pytest.raises(ValueError):  # unknown bundle
        DatasetFactoryConfig(bundle="nope", **common)
    with pytest.raises(ValueError):  # unknown representation
        DatasetFactoryConfig(representation="nope", **common)
    with pytest.raises(ValueError):  # unknown featurization
        DatasetFactoryConfig(
            representation="one_skeleton", featurization="nope", **common
        )
    with pytest.raises(ValueError):  # degree unsupported for bundles
        DatasetFactoryConfig(bundle="san", featurization="degree", **common)
    with pytest.raises(
        ValueError
    ):  # effective_resistance unsupported on graph
        DatasetFactoryConfig(
            representation="one_skeleton",
            featurization="effective_resistance",
            **common,
        )


def test_featurization_defaults_to_random(tmp_path):
    graph = DatasetFactoryConfig(
        root=str(tmp_path),
        dimension=2,
        variant="unbalanced",
        representation="one_skeleton",
    )
    bundle = DatasetFactoryConfig(
        root=str(tmp_path), dimension=2, variant="unbalanced", bundle="san"
    )
    assert graph.featurization == "random"
    assert bundle.featurization == "random"


def test_bundle_featurization_override(base_2d, tmp_path):
    # A bundle may override its featurization (here a propagated embedding).
    cfg = _cfg(
        tmp_path,
        base_2d,
        sub="sccnn-mc",
        dimension=2,
        variant="unbalanced",
        task="name",
        bundle="sccnn",
        featurization="moment_curve",
    )
    ds = build_dataset(cfg)
    assert {"x_0", "x_1", "x_2"}.issubset(set(ds[0].keys()))


def test_reconstruct_label_to_index_skips_unlabelled():
    import torch
    from torch_geometric.data import Data

    from mantra.factory.builder import _reconstruct_label_to_index

    labelled = Data(label="S^2", y=torch.tensor([0]))
    unlabelled = Data(num_nodes=1)  # no label/y -> skipped
    mapping = _reconstruct_label_to_index([labelled, unlabelled])
    assert mapping == {"S^2": 0}


def test_presets_reject_unknown_inputs():
    from mantra.factory.presets import (
        build_dataset_for_variant,
        bundle_required_keys,
    )

    with pytest.raises(ValueError):
        bundle_required_keys("nope", 2)
    with pytest.raises(ValueError):
        build_dataset_for_variant(
            "nope",
            dimension=2,
            root="x",
            version="latest",
            seed=42,
            name=None,
            pre_transform=None,
            transform=None,
            force_reload=False,
        )
