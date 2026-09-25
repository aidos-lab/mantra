"""Tests for the Pachner-walk OOD split of ``MantraDataset``."""

import pytest

from mantra.datasets import MantraDataset
from mantra.datasets.mantra_dataset import SubdivisionType
from mantra.utils.triangulation import Triangulation

from .conftest import manifold_entry
from .test_augmentation_invariants import is_orientable
from .test_mantra_divided import OCTAHEDRON
from .test_pachner_walks import TORUS, torus_entry

# The sources of one seed are drawn in a random order and with
# replacement, so (id, walk index) identifies a source between splits.


def make(make_manifolds_json, entries, tmp_path, **kwargs):
    path = make_manifolds_json(entries)
    kwargs.setdefault("dimension", 2)
    kwargs.setdefault("split_type", "ood")
    return MantraDataset(str(tmp_path / "root"), local_path=path, **kwargs)


@pytest.fixture
def entries():
    # Two homeomorphism types with distinct sizes: 5 octahedral spheres
    # (chi 2, 6 vertices) and 5 tori (chi 0, 7 vertices).
    return [
        manifold_entry(f"s{i}", triangulation=OCTAHEDRON, n_vertices=6)
        for i in range(5)
    ] + [torus_entry(f"t{i}") for i in range(5)]


def chi_and_orientable(data):
    t = Triangulation.from_list(data.triangulation)
    t.validate()
    return t.euler_characteristic(), is_orientable(data.triangulation)


class TestValidation:
    def test_target_and_match_are_exclusive(self, tmp_path):
        with pytest.raises(ValueError, match="not both"):
            MantraDataset(
                str(tmp_path / "root"),
                split_type="ood",
                division_type="pachner",
                target=16,
                match={"division_type": "graded", "graded_vertex_number": 16},
            )

    def test_match_needs_a_subdivision(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot match"):
            MantraDataset(
                str(tmp_path / "root"),
                split_type="ood",
                division_type="pachner",
                match={"division_type": "pachner"},
            )

    def test_move_weights_length_follows_dimension(self, tmp_path):
        with pytest.raises(ValueError, match="4 non-negative"):
            MantraDataset(
                str(tmp_path / "root"),
                split_type="ood",
                dimension=3,
                division_type="pachner",
                move_weights=(1, 1, 0),
            )

    def test_target_needs_a_vertex_adding_move(self, tmp_path):
        with pytest.raises(ValueError, match="vertex-adding"):
            MantraDataset(
                str(tmp_path / "root"),
                split_type="ood",
                division_type="pachner",
                target=16,
                move_weights=(1, 0, 0),
            )

    def test_target_below_source_raises(
        self, make_manifolds_json, entries, tmp_path
    ):
        with pytest.raises(ValueError, match="below the vertex count"):
            make(
                make_manifolds_json,
                entries,
                tmp_path,
                division_type="pachner",
                target=5,
            )


class TestWalks:
    def test_target_is_reached_exactly(
        self, make_manifolds_json, entries, tmp_path
    ):
        ds = make(
            make_manifolds_json,
            entries,
            tmp_path,
            division_type="pachner",
            target=12,
        )
        assert len(ds) > 0
        for d in ds:
            assert int(d.n_vertices) == 12
            assert len({v for s in d.triangulation for v in s}) == 12

    def test_walks_preserve_the_homeomorphism_type(
        self, make_manifolds_json, entries, tmp_path
    ):
        ds = make(
            make_manifolds_json,
            entries,
            tmp_path,
            division_type="pachner",
            target=14,
            mix=3,
            move_weights=(1, 1, 1),
        )
        expected = {"S^2": (2, True), "T^2": (0, True)}
        for d in ds:
            assert chi_and_orientable(d) == expected[d.name]

    def test_flip_walk_keeps_the_size(
        self, make_manifolds_json, entries, tmp_path
    ):
        ds = make(
            make_manifolds_json,
            entries,
            tmp_path,
            division_type="pachner",
            move_weights=(1, 0, 0),
            mix=5,
        )
        sizes = {d.name: int(d.n_vertices) for d in ds}
        assert sizes == {"S^2": 6, "T^2": 7}
        canonical = lambda t: sorted(sorted(s) for s in t)
        # The octahedron has flippable edges, so its flips change the
        # triangle set; the 7-vertex torus is neighborly (every vertex
        # pair is an edge), so no flip is possible and it passes through.
        for d in ds:
            source = OCTAHEDRON if d.name == "S^2" else TORUS
            changed = canonical(d.triangulation) != canonical(source)
            assert changed == (d.name == "S^2")

    def test_mix_changes_the_triangulation_not_the_size(
        self, make_manifolds_json, entries, tmp_path
    ):
        plain = make(
            make_manifolds_json,
            entries,
            tmp_path,
            division_type="pachner",
            target=12,
        )
        mixed = make(
            make_manifolds_json,
            entries,
            tmp_path / "mixed",
            division_type="pachner",
            target=12,
            mix=5,
        )
        assert [d.id for d in plain] == [d.id for d in mixed]
        assert all(int(d.n_vertices) == 12 for d in mixed)
        assert any(
            a.triangulation != b.triangulation for a, b in zip(plain, mixed)
        )


class TestMatchedSizes:
    @pytest.mark.parametrize(
        "match",
        [
            {"division_type": "graded", "graded_vertex_number": 11},
            {"division_type": "stellar", "fraction": 0.75},
            {"division_type": "stellar", "fraction": 1.0},
            {"division_type": "barycentric", "round": 1},
        ],
    )
    def test_matched_walks_have_the_sizes_of_the_subdivision(
        self, make_manifolds_json, entries, tmp_path, match
    ):
        subdivided = make(
            make_manifolds_json,
            entries,
            tmp_path,
            **{k: v for k, v in match.items()},
        )
        walked = make(
            make_manifolds_json,
            entries,
            tmp_path,
            division_type="pachner",
            match=match,
            mix=2,
        )
        # Same sources in the same order ...
        assert [d.id for d in subdivided] == [d.id for d in walked]
        # ... and the same vertex count entry by entry.
        assert [int(d.n_vertices) for d in subdivided] == [
            int(d.n_vertices) for d in walked
        ]
        assert (
            len({int(d.n_vertices) for d in walked}) > 1
            or match["division_type"] == "graded"
        )


class TestSharedSources:
    def test_every_subdivision_draws_the_same_sources(
        self, make_manifolds_json, entries, tmp_path
    ):
        variants = [
            dict(division_type="graded", graded_vertex_number=11),
            dict(division_type="stellar", fraction=0.75),
            dict(division_type="barycentric"),
            dict(division_type="pachner", target=11, mix=1),
        ]
        ids = [
            [d.id for d in make(make_manifolds_json, entries, tmp_path, **v)]
            for v in variants
        ]
        assert all(i == ids[0] for i in ids)
        # Base ids, i.e. the drawn sources, not only the ood tags.
        assert len(set(ids[0])) == len(ids[0])

    def test_cap_draws_the_same_sources_too(
        self, make_manifolds_json, entries, tmp_path
    ):
        kwargs = dict(
            max_ood_size_per_class=3, split_proportions=[0.2, 0.2, 0.6]
        )
        graded = make(
            make_manifolds_json,
            entries,
            tmp_path,
            division_type="graded",
            graded_vertex_number=11,
            **kwargs,
        )
        bary = make(
            make_manifolds_json,
            entries,
            tmp_path,
            division_type="barycentric",
            **kwargs,
        )
        assert [d.id for d in graded] == [d.id for d in bary]


class TestFileNames:
    def _name(self, **kwargs):
        obj = MantraDataset.__new__(MantraDataset)
        obj.division_type = SubdivisionType.PACHNER
        obj.max_ood_size_per_class = kwargs.pop("max_ood_size_per_class", None)
        obj.min_sample_per_class = None
        obj.split_proportions = [0.6, 0.2, 0.2]
        obj.stratified = False
        obj.kwargs = kwargs
        return obj.processed_file_names[-1]

    def test_target(self):
        assert self._name(target=16) == "ood_pachner_t16_w1-1-0_ss.pt"

    def test_match_and_mix(self):
        assert (
            self._name(
                match={"division_type": "stellar", "fraction": 0.75},
                mix=5,
                max_ood_size_per_class=100,
            )
            == "ood_pachner_m-stellar_0.75_w1-1-0_mix5_cap100_ss.pt"
        )

    def test_flip_only(self):
        assert (
            self._name(move_weights=(1, 0, 0), mix=5)
            == "ood_pachner_n_w1-0-0_mix5_ss.pt"
        )
