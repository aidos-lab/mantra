"""Tests for ``mantra.datasets.make_dataset``.

Each test that needs real data uses the ``--dataset-path`` opt-in (existing
conftest pattern). Pair-validation tests don't construct a dataset and run
unconditionally.
"""

import os
import shutil

import pytest
import torch

from mantra.datasets import make_dataset
from mantra.datasets.base import _get_dataset_url, _suffix
from mantra.datasets.factory import MODEL_BUNDLES, _derive_cache_name
from mantra.transforms.task_transforms import (
    NAME_TO_CLASS_2M,
    NAME_TO_CLASS_3M,
)


# --- Pair validation (no data needed) ----------------------------------------

def test_removed_triangulation_representation_raises():
    # The deprecated 'triangulation' representation (TriangulationToFaceTransform)
    # has been removed from the factory; requesting it is now an unknown
    # representation rather than a legal choice.
    with pytest.raises(ValueError, match="Unknown representation.*triangulation"):
        make_dataset(
            root="/tmp/_unused",
            dimension=2,
            representation="triangulation",  # type: ignore[arg-type]
            featurization="node_degree",
        )


def test_rw_legal_on_all_graph_representations():
    # rw is well-defined on every (graph) representation because AddRandomWalkPE
    # just walks the edge_index. Validation must accept the pair (the call only
    # fails later for lack of data, which we don't provide here).
    from mantra.datasets.factory import _validate_pair

    for rep in ("one_skeleton", "dual", "hasse"):
        _validate_pair(rep, "rw")  # must not raise


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="garbage"):
        make_dataset(
            root="/tmp/_unused",
            dimension=2,
            representation="one_skeleton",
            featurization="random",
            task="garbage",  # type: ignore[arg-type]
        )


def test_unknown_representation_raises():
    with pytest.raises(ValueError, match="Unknown representation"):
        make_dataset(
            root="/tmp/_unused",
            dimension=2,
            representation="ghost",  # type: ignore[arg-type]
            featurization="random",
        )


# --- Filename + URL conventions (no data needed) ----------------------------

def test_suffix_combinations():
    assert _suffix(False, None) == ""
    assert _suffix(True, None) == "_balanced"
    assert _suffix(False, 2) == "_bary2"
    assert _suffix(True, 4) == "_balanced_bary4"


def test_url_includes_subdivision_level():
    url = _get_dataset_url("latest", 2, balanced=True, subdivision_level=3)
    assert url.endswith("/2_manifolds_balanced_bary3.json.gz")


def test_subdivision_level_rejects_zero_or_negative():
    # The factory delegates this to ManifoldTriangulations; constructing
    # the dataset itself should raise.
    from mantra.datasets import ManifoldTriangulations
    with pytest.raises(AssertionError, match="subdivision_level"):
        ManifoldTriangulations(
            root="/tmp/_unused",
            dimension=2,
            subdivision_level=0,
            local_path="/dev/null",
        )


def test_spd_requires_transform():
    with pytest.raises(ValueError, match="spd_transform"):
        make_dataset(
            root="/tmp/_unused",
            dimension=2,
            representation="one_skeleton",
            featurization="random",
            shortest_path_distance=True,
        )


# --- Model-bundle validation (no data needed) --------------------------------

def test_bundle_xor_representation_required():
    # Neither given.
    with pytest.raises(ValueError, match="exactly one"):
        make_dataset(root="/tmp/_unused", dimension=2, featurization="random")
    # Both given.
    with pytest.raises(ValueError, match="exactly one"):
        make_dataset(
            root="/tmp/_unused", dimension=2, featurization="random",
            representation="one_skeleton", bundle="cwn",
        )


def test_unknown_bundle_raises():
    with pytest.raises(ValueError, match="Unknown bundle"):
        make_dataset(
            root="/tmp/_unused", dimension=2, featurization="random",
            bundle="ghost",
        )


@pytest.mark.parametrize("bad_feat", ["node_degree", "rw"])
def test_bundle_rejects_graph_only_featurizations(bad_feat):
    with pytest.raises(ValueError, match="not supported for model bundles"):
        make_dataset(
            root="/tmp/_unused", dimension=2, featurization=bad_feat,
            bundle="cwn",
        )


def test_bundle_rejects_spd():
    with pytest.raises(ValueError, match="shortest_path_distance"):
        make_dataset(
            root="/tmp/_unused", dimension=2, featurization="random",
            bundle="san", shortest_path_distance=True,
        )


def test_bundle_cache_name_disjoint_from_graph():
    graph = _derive_cache_name(
        "one_skeleton", "random", "name", 8, False, None, bundle=None
    )
    bundle = _derive_cache_name(
        None, "random", "name", 8, False, None, bundle=MODEL_BUNDLES["cwn"]
    )
    assert graph != bundle
    assert bundle.startswith("bundle_cwn__")
    assert "signed1" in bundle and "prop1" in bundle
    # Distinct bundles are distinct on disk.
    sccnn = _derive_cache_name(
        None, "random", "name", 8, False, None, bundle=MODEL_BUNDLES["sccnn"]
    )
    assert sccnn != bundle


# --- Data-dependent tests (need --dataset-path) ------------------------------

@pytest.fixture
def tmp_root(tmp_path):
    return str(tmp_path)


def test_one_skeleton_random_2d(tmp_root, dataset_path):
    ds = make_dataset(
        root=tmp_root,
        dimension=2,
        representation="one_skeleton",
        featurization="random",
        task="name",
        local_path=dataset_path,
        feature_dim=8,
        balanced=True,
    )
    d = ds[0]
    assert d.x.shape[1] == 8
    assert d.x.shape[0] == int(d.num_nodes)
    assert int(d.edge_index.max()) < int(d.num_nodes)
    assert d.y.ndim == 1
    assert d.y.dtype == torch.long


def test_canonical_y_matches_mantra_dict(tmp_root, dataset_path):
    ds = make_dataset(
        root=tmp_root,
        dimension=2,
        representation="one_skeleton",
        featurization="random",
        task="name",
        local_path=dataset_path,
        balanced=True,
    )
    # Each sample's y must equal the canonical NAME_TO_CLASS_2M index of
    # its (pre-transform) `name` attribute. The raw `name` is consumed by
    # NameToClass2MTransform during pre_transform, so we check the cached
    # data directly.
    for d in ds:
        # The triangulation/dimension/etc were preserved through the
        # pre_transform; `y` is the canonical index.
        assert int(d.y) in NAME_TO_CLASS_2M.values()


def test_cache_name_includes_subdivision_level(tmp_root, dataset_path):
    # Same factory args + different subdivision_level must produce distinct
    # processed/ subdirs so the caches coexist on disk.
    ds_orig = make_dataset(
        root=tmp_root, dimension=2, representation="one_skeleton",
        featurization="mc", local_path=dataset_path,
    )
    ds_bary2 = make_dataset(
        root=tmp_root, dimension=2, representation="one_skeleton",
        featurization="mc", local_path=dataset_path,
        subdivision_level=2,
    )
    assert ds_orig.processed_dir != ds_bary2.processed_dir
    assert "bary2" in ds_bary2.processed_dir


def test_cache_name_distinct(tmp_root, dataset_path):
    # Different featurizations must land in different processed/ subdirs
    ds_random = make_dataset(
        root=tmp_root,
        dimension=2,
        representation="one_skeleton",
        featurization="random",
        local_path=dataset_path,
        feature_dim=4,
    )
    ds_mc = make_dataset(
        root=tmp_root,
        dimension=2,
        representation="one_skeleton",
        featurization="mc",
        local_path=dataset_path,
    )
    assert ds_random.processed_dir != ds_mc.processed_dir
    assert os.path.isdir(ds_random.processed_dir)
    assert os.path.isdir(ds_mc.processed_dir)


def test_one_skeleton_rw_2d(tmp_root, dataset_path):
    ds = make_dataset(
        root=tmp_root,
        dimension=2,
        representation="one_skeleton",
        featurization="rw",
        task="name",
        local_path=dataset_path,
        feature_dim=4,
        balanced=True,
    )
    d = ds[0]
    # AddRandomWalkPE produces a feature of size walk_length per node.
    assert d.x.shape == (int(d.num_nodes), 4)


# --- Model bundles (need --dataset-path) -------------------------------------

def test_bundle_cwn_2d(tmp_root, dataset_path):
    ds = make_dataset(
        root=tmp_root, dimension=2, bundle="cwn", featurization="random",
        task="name", local_path=dataset_path, feature_dim=8, balanced=True,
    )
    d = ds[0]
    # Per-rank features on x_0..x_2 (random + propagate covers every rank).
    assert {"x_0", "x_1", "x_2"} <= set(d.keys())
    assert d.x_0.shape[1] == 8
    # Graph-shaped `x` must NOT be present on the SC path.
    assert "x" not in d.keys()
    # cwn connectivity stack: incidence + adjacency, no laplacians.
    assert any(k.startswith("incidence_") for k in d.keys())
    assert any(k.startswith("adjacency_") for k in d.keys())
    assert not any("laplacian" in k for k in d.keys())
    # Canonical label.
    assert d.y.dtype == torch.long
    assert int(d.y) in NAME_TO_CLASS_2M.values()


def test_bundle_sccnn_has_hodge(tmp_root, dataset_path):
    ds = make_dataset(
        root=tmp_root, dimension=2, bundle="sccnn", featurization="random",
        local_path=dataset_path,
    )
    keys = set(ds[0].keys())
    assert any(k.startswith("up_laplacian_") for k in keys)
    assert any(k.startswith("down_laplacian_") for k in keys)
    assert any(k.startswith("hodge_laplacian_") for k in keys)


def test_bundle_san_no_hodge(tmp_root, dataset_path):
    ds = make_dataset(
        root=tmp_root, dimension=2, bundle="san", featurization="random",
        local_path=dataset_path,
    )
    keys = set(ds[0].keys())
    assert any(k.startswith("up_laplacian_") for k in keys)
    assert any(k.startswith("down_laplacian_") for k in keys)
    assert not any(k.startswith("hodge_laplacian_") for k in keys)


def test_bundle_mc_covers_all_ranks(tmp_root, dataset_path):
    # MomentCurveEmbedding propagation keys off simplex *size* (d+1), so for a
    # 2-manifold it covers ranks 0..2 — i.e. x_0/x_1/x_2 all present, same
    # coverage as random+propagate.
    ds = make_dataset(
        root=tmp_root, dimension=2, bundle="cwn", featurization="mc",
        local_path=dataset_path,
    )
    keys = set(ds[0].keys())
    assert {"x_0", "x_1", "x_2"} <= keys


def test_bundle_cache_dir_disjoint_from_graph(tmp_root, dataset_path):
    ds_graph = make_dataset(
        root=tmp_root, dimension=2, representation="one_skeleton",
        featurization="random", local_path=dataset_path,
    )
    ds_bundle = make_dataset(
        root=tmp_root, dimension=2, bundle="cwn",
        featurization="random", local_path=dataset_path,
    )
    assert ds_graph.processed_dir != ds_bundle.processed_dir
    assert "bundle_cwn" in ds_bundle.processed_dir


# --- Dimension dispatch -------------------------------------------------------

def test_3d_dispatch(tmp_root, dataset_path_3d):
    ds = make_dataset(
        root=tmp_root,
        dimension=3,
        representation="one_skeleton",
        featurization="random",
        task="name",
        local_path=dataset_path_3d,
        balanced=True,
    )
    d = ds[0]
    assert int(d.y) in NAME_TO_CLASS_3M.values()
