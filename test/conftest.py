"""Shared fixtures for the dataset tests.

These helpers build *local* fixture files (a MANTRA-style JSON list and a
CY-style parquet) so the dataset classes can be exercised through their
``local_path`` code paths without ever touching the network.
"""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# A minimal 2-sphere triangulation (boundary of a tetrahedron).
SPHERE = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]


def manifold_entry(id, name="S^2", orientable=True, genus=0, **extra):
    """Build a single MANTRA-style manifold record."""
    entry = {
        "id": id,
        "triangulation": [list(s) for s in SPHERE],
        "dimension": 2,
        "n_vertices": 4,
        "betti_numbers": [1, 0, 1],
        "torsion_coefficients": ["", "", ""],
        "name": name,
        "genus": genus,
        "orientable": orientable,
        "vertex_transitive": True,
    }
    entry.update(extra)
    return entry


@pytest.fixture
def no_dedup(monkeypatch):
    """Disable isomorphism deduplication inside ``balance_dataset``.

    The shared fixtures reuse one tetrahedral sphere, so real
    deduplication would collapse them.
    """
    import mantra.augmentations.balancing as balancing_mod

    monkeypatch.setattr(balancing_mod, "find_duplicates", lambda *a, **k: [])


@pytest.fixture
def make_manifolds_json(tmp_path):
    """Return a factory writing a list of entries to a JSON file."""

    def _make(entries, filename="manifolds.json"):
        path = tmp_path / filename
        path.write_text(json.dumps(entries))
        return str(path)

    return _make


@pytest.fixture
def balanced_entries():
    """A small, class-balanced 2D dataset (5 orientable, 5 not)."""
    return [
        manifold_entry(f"m{i}", name="S^2", orientable=True, genus=0)
        for i in range(5)
    ] + [
        manifold_entry(f"m{i + 5}", name="RP^2", orientable=False, genus=0)
        for i in range(5)
    ]


@pytest.fixture
def pairwise_entries():
    """A 2D dataset mixing the ``S^2``/``T^2`` comparison pair with ``RP^2``.

    The ``RP^2`` record exercises the exclusion branch of the pairwise
    comparison (it falls outside the comparison pair).
    """
    return (
        [
            manifold_entry(f"s{i}", name="S^2", orientable=True)
            for i in range(2)
        ]
        + [
            manifold_entry(f"t{i}", name="T^2", orientable=True)
            for i in range(2)
        ]
        + [manifold_entry("r0", name="RP^2", orientable=False)]
    )


@pytest.fixture
def make_cy_parquet(tmp_path):
    """Return a factory writing CY-style rows to a parquet file.

    ``rows`` is a list of ``(simplices, vertices)`` tuples.
    """

    def _make(rows, filename="manifolds.parquet"):
        path = tmp_path / filename
        df = pd.DataFrame(
            {
                "simplices": [r[0] for r in rows],
                "vertices": [r[1] for r in rows],
            }
        )
        pq.write_table(pa.Table.from_pandas(df), str(path))
        return str(path)

    return _make


@pytest.fixture
def cy_rows():
    """Two tetrahedral CY-style rows (4 vertices in 3-space)."""
    simplices = [[0, 1, 2, 3]]
    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return [(simplices, vertices), (simplices, vertices)]
