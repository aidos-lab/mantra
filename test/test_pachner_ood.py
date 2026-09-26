"""Tests for the Pachner-walk OOD split and the shared OOD sources."""

import pytest

from mantra.datasets import MantraDataset
from mantra.datasets.mantra_dataset import SubdivisionType
from mantra.utils.triangulation import Triangulation

from .conftest import manifold_entry
from .test_augmentation_invariants import is_orientable
from .test_mantra_divided import OCTAHEDRON
from .test_pachner_walks import TORUS, torus_entry

# 0.2/0.2/0.6 puts 6 of the 10 entries into the test split, three per
# class, so the source draw (with replacement) can actually differ
# between two OOD builds and the tests below are not vacuous.
SPLIT = [0.2, 0.2, 0.6]


def make(make_manifolds_json, entries, tmp_path, **kwargs):
    path = make_manifolds_json(entries)
    kwargs.setdefault("dimension", 2)
    kwargs.setdefault("split_type", "ood")
    kwargs.setdefault("split_proportions", SPLIT)
    return MantraDataset(str(tmp_path / "root"), local_path=path, **kwargs)


@pytest.fixture
def entries():
    # Two homeomorphism types of distinct sizes: 5 octahedral spheres
    # (chi 2, 6 vertices) and 5 tori (chi 0, 7 vertices).
    return [
        manifold_entry(f"s{i}", triangulation=OCTAHEDRON, n_vertices=6)
        for i in range(5)
    ] + [torus_entry(f"t{i}") for i in range(5)]


def chi_and_orientable(data):
    t = Triangulation.from_list(data.triangulation)
    t.validate()
    return t.euler_characteristic(), is_orientable(data.triangulation)


def canonical(triangulation):
    return sorted(sorted(s) for s in triangulation)


def source_ids(dataset):
    return [d.id.rsplit("_ood_", 1)[0] for d in dataset]


class TestValidation:
    def new(self, tmp_path, **kwargs):
        return MantraDataset(
            str(tmp_path / "root"), split_type="ood", **kwargs
        )

    def test_unknown_argument_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not take \\['targt'\\]"):
            self.new(tmp_path, division_type="pachner", targt=16)
        with pytest.raises(ValueError, match="does not take \\['fraction'\\]"):
            self.new(
                tmp_path,
                division_type="graded",
                graded_vertex_number=16,
                fraction=1,
            )

    def test_target_and_match_are_exclusive(self, tmp_path):
        with pytest.raises(ValueError, match="not both"):
            self.new(
                tmp_path,
                division_type="pachner",
                target=16,
                match={"division_type": "graded", "graded_vertex_number": 16},
            )

    def test_match_is_validated_like_the_subdivision(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot match"):
            self.new(
                tmp_path,
                division_type="pachner",
                match={"division_type": "pachner"},
            )
        with pytest.raises(
            ValueError, match="requires 'graded_vertex_number'"
        ):
            self.new(
                tmp_path,
                division_type="pachner",
                match={"division_type": "graded"},
            )
        with pytest.raises(ValueError, match="does not take \\['fracton'\\]"):
            self.new(
                tmp_path,
                division_type="pachner",
                match={"division_type": "stellar", "fracton": 0.5},
            )
        with pytest.raises(ValueError, match="strictly greater"):
            self.new(
                tmp_path,
                division_type="pachner",
                max_vertices=16,
                match={"division_type": "graded", "graded_vertex_number": 16},
            )

    def test_target_must_be_an_int_above_max_vertices(self, tmp_path):
        with pytest.raises(ValueError, match="must be an int"):
            self.new(tmp_path, division_type="pachner", target=12.7)
        with pytest.raises(ValueError, match="strictly greater"):
            self.new(
                tmp_path, division_type="pachner", target=10, max_vertices=10
            )

    def test_move_weights(self, tmp_path):
        with pytest.raises(ValueError, match="three non-negative"):
            self.new(tmp_path, division_type="pachner", move_weights=(1, 1))
        with pytest.raises(ValueError, match="1-3 move"):
            self.new(
                tmp_path,
                division_type="pachner",
                target=16,
                move_weights=(1, 0, 0),
            )

    def test_three_dimensions_are_not_supported(self, tmp_path):
        with pytest.raises(NotImplementedError):
            self.new(tmp_path, dimension=3, division_type="pachner", target=9)

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
        assert len(ds) == 6
        for d in ds:
            assert int(d.n_vertices) == 12
            assert len({v for s in d.triangulation for v in s}) == 12

    def test_walks_preserve_the_homeomorphism_type(
        self, make_manifolds_json, entries, tmp_path
    ):
        # chi and orientability classify closed surfaces, so together
        # they are a complete check; coarsening moves are on as well.
        ds = make(
            make_manifolds_json,
            entries,
            tmp_path,
            division_type="pachner",
            target=14,
            mix=3,
            move_weights=(1, 1, 1),
            relabel=True,
        )
        expected = {"S^2": (2, True), "T^2": (0, True)}
        for d in ds:
            assert chi_and_orientable(d) == expected[d.name]

    def test_flip_only_walk_keeps_the_size(
        self, make_manifolds_json, entries, tmp_path
    ):
        ds = make(
            make_manifolds_json,
            entries,
            tmp_path,
            division_type="pachner",
            mix=5,
        )
        assert {d.name: int(d.n_vertices) for d in ds} == {"S^2": 6, "T^2": 7}
        # The octahedron has flippable edges, so its triangle set changes;
        # the 7-vertex torus is neighborly (every vertex pair is an edge),
        # so no flip exists and it passes through unchanged.
        for d in ds:
            source = OCTAHEDRON if d.name == "S^2" else TORUS
            assert (canonical(d.triangulation) != canonical(source)) == (
                d.name == "S^2"
            )

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
        subdivided = make(make_manifolds_json, entries, tmp_path, **match)
        walked = make(
            make_manifolds_json,
            entries,
            tmp_path,
            division_type="pachner",
            match=match,
            mix=2,
        )
        # Same sources in the same order, same vertex count entry by entry.
        assert source_ids(subdivided) == source_ids(walked)
        assert [int(d.n_vertices) for d in subdivided] == [
            int(d.n_vertices) for d in walked
        ]
        # And the walk is not the subdivision.
        assert any(
            a.triangulation != b.triangulation
            for a, b in zip(subdivided, walked)
        )


class TestSharedSources:
    VARIANTS = [
        dict(division_type="graded", graded_vertex_number=11),
        dict(division_type="stellar", fraction=0.75),
        dict(division_type="barycentric"),
        dict(division_type="pachner", target=11, mix=1),
        dict(division_type="pachner", relabel=True),
    ]

    def test_every_subdivision_draws_the_same_sources(
        self, make_manifolds_json, entries, tmp_path
    ):
        ids = [
            source_ids(make(make_manifolds_json, entries, tmp_path, **v))
            for v in self.VARIANTS
        ]
        assert all(i == ids[0] for i in ids)
        # With replacement, from three candidates per class: the draw is
        # a real one, not the list of candidates.
        assert len(ids[0]) == 6

    def test_cap_draws_the_same_sources_too(
        self, make_manifolds_json, entries, tmp_path
    ):
        ids = [
            source_ids(
                make(
                    make_manifolds_json,
                    entries,
                    tmp_path,
                    max_ood_size_per_class=2,
                    **v,
                )
            )
            for v in self.VARIANTS[:3]
        ]
        assert all(i == ids[0] for i in ids) and len(ids[0]) == 4

    def test_barycentric_sources_are_pinned(
        self, make_manifolds_json, entries, tmp_path
    ):
        # Regression pin (seed 42): the draw-free subdivisions keep the
        # sources they had before the shared draw (value taken from the
        # code before it).
        ds = make(
            make_manifolds_json, entries, tmp_path, division_type="barycentric"
        )
        assert source_ids(ds) == ["s0", "s1", "s1", "t3", "t2", "t2"]


class TestRelabel:
    def test_relabel_only_is_an_isomorphic_copy(
        self, make_manifolds_json, entries, tmp_path
    ):
        ds = make(
            make_manifolds_json,
            entries,
            tmp_path,
            division_type="pachner",
            relabel=True,
        )
        changed = 0
        for d in ds:
            source = OCTAHEDRON if d.name == "S^2" else TORUS
            assert int(d.n_vertices) == len({v for s in source for v in s})
            assert len(d.triangulation) == len(source)
            # A relabelling is a bijection on the vertices that maps the
            # triangle set onto the source's: find it and check it.
            for perm in _relabellings(d.triangulation, source):
                break
            else:
                pytest.fail(f"{d.id} is not a relabelling of its source")
            changed += canonical(d.triangulation) != canonical(source)
        assert changed > 0

    def test_relabel_vertices(self):
        t = Triangulation.from_list(OCTAHEDRON)
        perm = {1: 2, 2: 1, 3: 5, 4: 4, 5: 3, 6: 6}
        t.relabel_vertices(perm)
        assert t.to_list() == canonical(
            [[perm[v] for v in s] for s in OCTAHEDRON]
        )
        with pytest.raises(ValueError, match="bijection"):
            t.relabel_vertices({1: 1})

    def test_relabel_names_the_cache(self):
        obj = MantraDataset.__new__(MantraDataset)
        obj.division_type = SubdivisionType.PACHNER
        obj.relabel = True
        obj.max_ood_size_per_class = None
        obj.min_sample_per_class = None
        obj.split_proportions = [0.6, 0.2, 0.2]
        obj.stratified = False
        obj.kwargs = {}
        assert obj.processed_file_names[-1] == "ood_pachner_n_rl_ss.pt"


def _relabellings(triangulation, source):
    """Yield vertex bijections mapping ``triangulation`` onto ``source``."""
    from itertools import permutations

    verts = sorted({v for s in triangulation for v in s})
    target = {frozenset(s) for s in source}
    for image in permutations(sorted({v for s in source for v in s})):
        perm = dict(zip(verts, image))
        if {frozenset(perm[v] for v in s) for s in triangulation} == target:
            yield perm


class TestFileNames:
    def _name(self, **kwargs):
        obj = MantraDataset.__new__(MantraDataset)
        obj.division_type = SubdivisionType.PACHNER
        obj.relabel = kwargs.pop("relabel", False)
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

    def test_flip_only_has_no_weights(self):
        assert self._name(mix=5) == "ood_pachner_n_mix5_ss.pt"
