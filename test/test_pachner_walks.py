"""Tests for ``mantra.datasets.pachner_walks``."""

from itertools import combinations

import pytest

from torch_geometric.data import Batch

from mantra.datasets import MantraDataset, PachnerWalkDataset
from mantra.utils.constants import TORUS_TRIANGULATION_MINUS_FACE
from mantra.utils.triangulation import Triangulation

from .conftest import manifold_entry

TORUS = TORUS_TRIANGULATION_MINUS_FACE + [[1, 2, 3]]
# Boundary of the 4-simplex: the minimal 3-sphere.
S3 = [list(c) for c in combinations(range(1, 6), 4)]


def torus_entry(id):
    return manifold_entry(
        id,
        name="T^2",
        genus=1,
        triangulation=TORUS,
        n_vertices=7,
        betti_numbers=[1, 2, 1],
    )


def s3_entry(id):
    return {
        "id": id,
        "triangulation": S3,
        "dimension": 3,
        "n_vertices": 5,
        "betti_numbers": [1, 0, 0, 1],
        "torsion_coefficients": ["", "", "", ""],
        "name": "S^3",
        "vertex_transitive": True,
    }


@pytest.fixture
def entries_2d():
    return [manifold_entry(f"s{i}") for i in range(5)] + [
        torus_entry(f"t{i}") for i in range(5)
    ]


@pytest.fixture
def entries_3d():
    return [s3_entry(f"s{i}") for i in range(10)]


def make_walks(make_manifolds_json, entries, tmp_path, **kwargs):
    path = make_manifolds_json(entries)
    kwargs.setdefault("dimension", 2)
    return PachnerWalkDataset(
        str(tmp_path / "root"), local_path=path, **kwargs
    )


def make_plain(make_manifolds_json, entries, tmp_path, **kwargs):
    path = make_manifolds_json(entries)
    kwargs.setdefault("dimension", 2)
    return MantraDataset(str(tmp_path / "root"), local_path=path, **kwargs)


class TestValidation:
    def test_negative_walk_length_raises(self, tmp_path):
        with pytest.raises(ValueError, match="walk_length"):
            PachnerWalkDataset(
                str(tmp_path / "root"), split_type="train", walk_length=-1
            )

    def test_zero_moves_per_step_raises(self, tmp_path):
        with pytest.raises(ValueError, match="moves_per_step"):
            PachnerWalkDataset(
                str(tmp_path / "root"), split_type="train", moves_per_step=0
            )


class TestNoWalk:
    def test_matches_mantra_dataset(
        self, make_manifolds_json, entries_2d, tmp_path
    ):
        for split in ["train", "val", "test", "ood"]:
            plain = make_plain(
                make_manifolds_json, entries_2d, tmp_path, split_type=split
            )
            walks = make_walks(
                make_manifolds_json, entries_2d, tmp_path, split_type=split
            )
            assert [d.id for d in walks] == [d.id for d in plain]
            assert walks.processed_file_names == plain.processed_file_names
            assert not hasattr(walks[0], "walk_step")


class TestWalks:
    @pytest.mark.parametrize("dimension", [2, 3])
    def test_splits_are_expanded(
        self, make_manifolds_json, entries_2d, entries_3d, tmp_path, dimension
    ):
        entries = entries_2d if dimension == 2 else entries_3d
        walk_length, moves_per_step = 3, 2
        for split in ["train", "val", "test"]:
            plain = make_plain(
                make_manifolds_json,
                entries,
                tmp_path,
                split_type=split,
                dimension=dimension,
            )
            walks = make_walks(
                make_manifolds_json,
                entries,
                tmp_path,
                split_type=split,
                dimension=dimension,
                walk_length=walk_length,
                moves_per_step=moves_per_step,
            )
            assert len(walks) == (walk_length + 1) * len(plain)

            n_vertices_changed = False
            for i, data in enumerate(walks):
                walk_base, step = divmod(i, walk_length + 1)
                base = plain[walk_base]
                assert int(data.walk_base) == walk_base
                assert int(data.walk_step) == step
                expected_id = (
                    base.id if step == 0 else f"{base.id}_walk_{step}"
                )
                assert data.id == expected_id
                assert data.name == base.name
                assert data.betti_numbers == base.betti_numbers
                assert int(data.dimension) == dimension
                if dimension == 2:
                    assert bool(data.orientable) == bool(base.orientable)
                    assert int(data.genus) == int(base.genus)

                tri = Triangulation.from_list(data.triangulation)
                tri.validate()
                assert tri.n_vertices == int(data.n_vertices)
                assert tri.euler_characteristic() == (
                    Triangulation.from_list(
                        base.triangulation
                    ).euler_characteristic()
                )
                if step == 0:
                    assert data.triangulation == base.triangulation
                elif int(data.n_vertices) != int(base.n_vertices):
                    n_vertices_changed = True
            assert n_vertices_changed

    def test_ood_split_is_not_expanded(
        self, make_manifolds_json, entries_2d, tmp_path
    ):
        kwargs = dict(division_type="barycentric")
        plain = make_plain(
            make_manifolds_json,
            entries_2d,
            tmp_path,
            split_type="ood",
            **kwargs,
        )
        walks = make_walks(
            make_manifolds_json,
            entries_2d,
            tmp_path,
            split_type="ood",
            walk_length=3,
            **kwargs,
        )
        assert [d.id for d in walks] == [d.id for d in plain]
        assert [d.triangulation for d in walks] == [
            d.triangulation for d in plain
        ]
        assert not hasattr(walks[0], "walk_step")

    def test_walks_are_reproducible_and_seed_dependent(
        self, make_manifolds_json, entries_2d, tmp_path
    ):
        def triangulations(walk_seed):
            ds = make_walks(
                make_manifolds_json,
                entries_2d,
                tmp_path,
                split_type="train",
                walk_length=3,
                walk_seed=walk_seed,
            )
            return [d.triangulation for d in ds]

        assert triangulations(1) == triangulations(1)
        assert triangulations(1) != triangulations(2)

    def test_pre_transform_sees_expanded_entries(
        self, make_manifolds_json, entries_2d, tmp_path
    ):
        def pre_transform(d):
            d.tagged_step = int(getattr(d, "walk_step", -1))
            return d

        ds = make_walks(
            make_manifolds_json,
            entries_2d,
            tmp_path,
            split_type="train",
            walk_length=2,
            pre_transform=pre_transform,
        )
        assert [int(d.tagged_step) for d in ds] == [0, 1, 2] * 6


class TestCache:
    def test_file_names_encode_walk_parameters(
        self, make_manifolds_json, entries_2d, tmp_path
    ):
        names = {}
        for key, kwargs in {
            "none": {},
            "walk": dict(walk_length=3),
            "walk_other_seed": dict(walk_length=3, walk_seed=7),
            "walk_more_moves": dict(walk_length=3, moves_per_step=2),
        }.items():
            ds = make_walks(
                make_manifolds_json,
                entries_2d,
                tmp_path,
                split_type="train",
                **kwargs,
            )
            names[key] = ds.processed_file_names
        assert len({tuple(n[:3]) for n in names.values()}) == 4
        assert len({n[3] for n in names.values()}) == 1
        assert names["walk"][0] == "train_walk3x1_ws42.pt"
        assert names["walk_other_seed"][0] == "train_walk3x1_ws7.pt"
        assert names["walk_more_moves"][0] == "train_walk3x2_ws42.pt"

    def test_second_instantiation_loads_from_cache(
        self, make_manifolds_json, entries_2d, tmp_path, monkeypatch
    ):
        first = make_walks(
            make_manifolds_json,
            entries_2d,
            tmp_path,
            split_type="val",
            walk_length=2,
        )
        monkeypatch.setattr(
            PachnerWalkDataset,
            "process",
            lambda self: pytest.fail("cache was not used"),
        )
        second = make_walks(
            make_manifolds_json,
            entries_2d,
            tmp_path,
            split_type="val",
            walk_length=2,
        )
        assert [d.id for d in second] == [d.id for d in first]
        assert [d.triangulation for d in second] == [
            d.triangulation for d in first
        ]


def test_walk_metadata_survives_collation(
    make_manifolds_json, entries_2d, tmp_path
):
    """The walk attributes must not be renumbered by the PyG collate.

    ``Data.__inc__`` increments every attribute whose key contains
    ``index``, which would silently merge distinct walks into one.
    """
    dataset = make_walks(
        make_manifolds_json,
        entries_2d,
        tmp_path,
        split_type="train",
        walk_length=2,
        min_sample_per_class=1,
    )
    entries = list(dataset)
    batch = Batch.from_data_list(entries)

    assert batch.walk_base.tolist() == [int(d.walk_base) for d in entries]
    assert batch.walk_step.tolist() == [int(d.walk_step) for d in entries]
