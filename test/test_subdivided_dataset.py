"""End-to-end tests for ``mantra.datasets.subdivided.MANTRASubdivided``.

These exercise the subdivision-derivation pipeline through the dataset's
``local_path`` (and monkeypatched download) code paths, with no network. The
expected vertex/cell counts are exact, classical facts about barycentric and
stellar subdivision, so they double as verified-output checks.
"""

import json
import os

import pytest

import mantra.datasets.subdivided as subdivided_mod
from mantra.datasets import MANTRASubdivided


def _entry(triangulation, name="S^2", orientable=True, dimension=2):
    verts = sorted({v for s in triangulation for v in s})
    return {
        "id": "x",
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
def write_base(tmp_path):
    """Write a base (unsubdivided) MANTRA JSON, returning its path."""

    def _write(entries, filename="base.json"):
        path = tmp_path / filename
        path.write_text(json.dumps(entries))
        return str(path)

    return _write


TRIANGLE = [[1, 2, 3]]
TETRAHEDRON = [[1, 2, 3, 4]]
# Boundary of a tetrahedron: a 4-vertex 2-sphere with four triangles.
SPHERE_2D = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]


def _build(root, dimension, variant, kwargs, base_path):
    return MANTRASubdivided(
        str(root),
        dimension=dimension,
        variant=variant,
        subdivision_kwargs=kwargs,
        local_path=base_path,
    )


# --------------------------------------------------------------------------- #
# Exact subdivision geometry
# --------------------------------------------------------------------------- #


def test_triangle_barycentric_counts(write_base, tmp_path):
    base = write_base([_entry(TRIANGLE)])
    ds = _build(
        tmp_path / "r",
        2,
        "barycentric",
        {"mode": "full_barycentric", "n_levels": 1, "seed": 42},
        base,
    )
    # A barycentrically subdivided triangle has 6 cells and 7 vertices.
    assert len(ds[0].triangulation) == 6
    assert int(ds[0].n_vertices) == 7


def test_triangle_stellar_full_counts(write_base, tmp_path):
    base = write_base([_entry(TRIANGLE)])
    ds = _build(
        tmp_path / "r",
        2,
        "stellar_full",
        {"mode": "stellar", "n_levels": 1.0, "seed": 42},
        base,
    )
    # A 1->3 stellar move on a triangle: 3 cells, 4 vertices.
    assert len(ds[0].triangulation) == 3
    assert int(ds[0].n_vertices) == 4


def test_tetrahedron_barycentric_counts(write_base, tmp_path):
    base = write_base([_entry(TETRAHEDRON, dimension=3)])
    ds = _build(
        tmp_path / "r",
        3,
        "barycentric",
        {"mode": "full_barycentric", "n_levels": 1, "seed": 42},
        base,
    )
    # A barycentrically subdivided tetrahedron has 24 cells and 15 vertices.
    assert len(ds[0].triangulation) == 24
    assert int(ds[0].n_vertices) == 15


def test_tetrahedron_stellar_full_counts(write_base, tmp_path):
    base = write_base([_entry(TETRAHEDRON, dimension=3)])
    ds = _build(
        tmp_path / "r",
        3,
        "stellar_full",
        {"mode": "stellar", "n_levels": 1.0, "seed": 42},
        base,
    )
    # A 1->4 stellar move on a tetrahedron: 4 cells, 5 vertices.
    assert len(ds[0].triangulation) == 4
    assert int(ds[0].n_vertices) == 5


# --------------------------------------------------------------------------- #
# Fractional stellar + graded
# --------------------------------------------------------------------------- #


def test_stellar_fraction_is_deterministic(write_base, tmp_path):
    base = write_base([_entry(SPHERE_2D)])
    kwargs = {"mode": "stellar", "n_levels": 0.75, "seed": 42}
    a = _build(tmp_path / "a", 2, "stellar_0.75", kwargs, base)
    b = _build(tmp_path / "b", 2, "stellar_0.75", kwargs, base)
    assert a[0].triangulation == b[0].triangulation
    # 0.75 of 4 triangles -> 3 subdivided (each +3 cells, +1 vertex).
    assert int(a[0].n_vertices) == 4 + 3
    assert len(a[0].triangulation) == 4 + 3 * (3 - 1)


def test_graded_reaches_target_and_is_deterministic(write_base, tmp_path):
    base = write_base([_entry(TRIANGLE) for _ in range(3)])
    kwargs = {"mode": "graded", "n_levels": 8, "n_smallest": 2, "seed": 42}
    a = _build(tmp_path / "a", 2, "graded", kwargs, base)
    b = _build(tmp_path / "b", 2, "graded", kwargs, base)
    # n_smallest=2 cohort per class, each grown to exactly 8 vertices.
    assert len(a) == 2
    assert all(int(d.n_vertices) == 8 for d in a)
    assert [d.triangulation for d in a] == [d.triangulation for d in b]


# --------------------------------------------------------------------------- #
# Naming, caching and error handling
# --------------------------------------------------------------------------- #


def test_disambiguated_file_names(write_base, tmp_path):
    base = write_base([_entry(TRIANGLE)])
    ds = _build(
        tmp_path / "r",
        2,
        "barycentric",
        {"mode": "full_barycentric", "n_levels": 1, "seed": 42},
        base,
    )
    assert ds.raw_file_names == ["2_manifolds_barycentric.json"]
    assert ds.processed_file_names == ["data_2_barycentric.pt"]


def test_multiple_levels_rejected(write_base, tmp_path):
    base = write_base([_entry(TRIANGLE)])
    with pytest.raises(ValueError, match="exactly one level"):
        _build(
            tmp_path / "r",
            2,
            "barycentric",
            {"mode": "full_barycentric", "n_levels": 2, "seed": 42},
            base,
        )


def test_add_version_to_root_branches():
    obj = MANTRASubdivided.__new__(MANTRASubdivided)
    obj.dimension = 2
    obj.version = "v1.0.0"
    assert obj._add_version_to_root() == "/mantra/v1.0.0/2D"
    obj.version = "latest"
    assert obj._add_version_to_root() == "/mantra/2D"


def test_download_branch_then_subdivides(write_base, tmp_path, monkeypatch):
    entries = [_entry(TRIANGLE)]
    content = json.dumps(entries)

    def fake_download_url(url, scratch_dir):
        gz = os.path.join(scratch_dir, "2_manifolds.json.gz")
        with open(gz, "w") as f:
            f.write("dummy")
        fake_download_url.url = url
        return gz

    def fake_extract_gz(path, scratch_dir):
        with open(os.path.join(scratch_dir, "2_manifolds.json"), "w") as f:
            f.write(content)

    monkeypatch.setattr(subdivided_mod, "download_url", fake_download_url)
    monkeypatch.setattr(subdivided_mod, "extract_gz", fake_extract_gz)

    ds = MANTRASubdivided(
        str(tmp_path / "r"),
        dimension=2,
        variant="barycentric",
        subdivision_kwargs={
            "mode": "full_barycentric",
            "n_levels": 1,
            "seed": 42,
        },
        version="latest",
    )
    # The base (unbalanced) artifact was fetched and then subdivided.
    assert fake_download_url.url.endswith("2_manifolds.json.gz")
    assert len(ds[0].triangulation) == 6
