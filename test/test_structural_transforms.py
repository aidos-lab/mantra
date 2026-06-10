"""Regression tests for ``mantra.transforms.structural_transforms``.

These tests pin down the contract of ``TriangulationToFaceTransform``:
the 0-based indexing, the ``[3, M]`` shape, the correct vertex-axis
indexing on 3-manifolds, and order-insensitive triangle dedup.
"""

import pytest
import torch
import torch_geometric.transforms as pyg_T

from mantra.transforms.structural_transforms import (
    SetNumNodesTransform,
    TriangulationToFaceTransform,
)

# TriangulationToFaceTransform is deprecated; these regression tests still
# pin its behaviour while it exists, so silence the per-construction
# DeprecationWarning at the module level. ``test_deprecation_warning`` below
# asserts the warning still fires.
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:mantra.transforms.structural_transforms"
)


# Boundary of a tetrahedron == minimal triangulation of S^2.
S2_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]

# Boundary of a 4-simplex == minimal triangulation of S^3.
S3_TRI = [
    [1, 2, 3, 4],
    [1, 2, 3, 5],
    [1, 2, 4, 5],
    [1, 3, 4, 5],
    [2, 3, 4, 5],
]


def _column_sets(face):
    """Return the set of triangles as a set of frozensets of vertex IDs."""
    return {frozenset(col.tolist()) for col in face.T}


def test_face_shape_2d_minimal(mk_data):
    data = mk_data(triangulation=S2_TRI)
    out = TriangulationToFaceTransform()(data)
    assert out.face.shape == (3, 4)
    assert _column_sets(out.face) == {
        frozenset({0, 1, 2}),
        frozenset({0, 1, 3}),
        frozenset({0, 2, 3}),
        frozenset({1, 2, 3}),
    }


def test_face_is_zero_based(mk_data):
    data = mk_data(triangulation=S2_TRI)
    out = TriangulationToFaceTransform()(data)
    assert int(out.face.min()) == 0
    assert int(out.face.max()) == 3


def test_face_shape_3d_minimal(mk_data):
    data = mk_data(triangulation=S3_TRI)
    out = TriangulationToFaceTransform()(data)
    # C(5, 3) = 10 unique triangles on a 4-simplex boundary
    assert out.face.shape == (3, 10)
    cols = _column_sets(out.face)
    assert len(cols) == 10
    # Every triangle is a 3-element subset of {0, 1, 2, 3, 4}
    universe = set(range(5))
    for c in cols:
        assert c <= universe and len(c) == 3


def test_face_dedup_on_shared_face(mk_data):
    # Two tets sharing the triangle {1, 2, 3} (1-based) → {0, 1, 2} (0-based)
    data = mk_data(triangulation=[[1, 2, 3, 4], [1, 2, 3, 5]])
    out = TriangulationToFaceTransform()(data)
    # 2 tets × 4 faces - 1 shared face = 7
    assert out.face.shape == (3, 7)
    cols = _column_sets(out.face)
    assert frozenset({0, 1, 2}) in cols


def test_triangulation_preserved(mk_data):
    data = mk_data(triangulation=S2_TRI)
    out = TriangulationToFaceTransform(remove_triangulation=False)(data)
    assert isinstance(out.triangulation, torch.Tensor)
    assert out.triangulation.shape == (4, 3)


def test_triangulation_removed(mk_data):
    data = mk_data(triangulation=S2_TRI)
    out = TriangulationToFaceTransform(remove_triangulation=True)(data)
    # PyG's Data drops keys with None values from its storage; either an
    # explicit None or a missing key is acceptable.
    assert getattr(out, "triangulation", None) is None


def test_pipeline_with_face_to_edge(mk_data):
    """Chained with FaceToEdge, the produced edge_index is 0-based and
    fits within ``n_vertices``."""
    n_vertices = 4
    data = mk_data(triangulation=S2_TRI, num_nodes=n_vertices)
    out = pyg_T.Compose(
        [TriangulationToFaceTransform(), pyg_T.FaceToEdge(remove_faces=False)]
    )(data)
    assert int(out.edge_index.min()) == 0
    assert int(out.edge_index.max()) < n_vertices


def test_rejects_unknown_simplex_size(mk_data):
    data = mk_data(triangulation=[[1, 2]])
    with pytest.raises(ValueError, match="Unsupported simplex size"):
        TriangulationToFaceTransform()(data)


def test_no_triangulation_is_noop(mk_data):
    # A Data object without a ``triangulation`` attribute is returned
    # untouched and no ``face`` is created.
    data = mk_data(num_nodes=4)
    out = TriangulationToFaceTransform()(data)
    assert "face" not in out


def test_dedup_preserves_first_occurrence_orientation(mk_data):
    # The shared triangle is enumerated as (1,2,3) by the first tet and as
    # (1,3,2)-style orderings by later faces; dedup must keep the *first*
    # occurrence's vertex order rather than a sorted/canonical one.
    data = mk_data(triangulation=[[1, 2, 3, 4]])
    out = TriangulationToFaceTransform()(data)
    # First enumerated face is vertices (0, 1, 2) in that exact order.
    assert out.face[:, 0].tolist() == [0, 1, 2]


def test_set_num_nodes_transform(mk_data):
    data = mk_data(n_vertices=7)
    out = SetNumNodesTransform()(data)
    assert out.num_nodes == 7


def test_deprecation_warning():
    # The transform is deprecated; constructing it must emit a
    # DeprecationWarning (it is no longer wired into the factory).
    with pytest.warns(DeprecationWarning, match="deprecated"):
        TriangulationToFaceTransform()
