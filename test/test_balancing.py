"""Tests for ``mantra.augmentations.balancing``."""

import random

from mantra.augmentations import balancing
from mantra.augmentations.balancing import (
    _augment_triangulation,
    _augment_with_topology_change,
    _downsample,
    _find_topology_sources,
    _normalize_name,
    balance_dataset,
    print_statistics,
)

SPHERE = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
# A closed-ish 3D seed: two tetrahedra sharing a face.
TET_PAIR = [[1, 2, 3, 4], [1, 2, 3, 5]]


def sphere_entry(id="s0", name="S^2", nv=4, **extra):
    e = {
        "id": id,
        "name": name,
        "n_vertices": nv,
        "triangulation": [list(s) for s in SPHERE],
        "betti_numbers": [1, 0, 1],
        "orientable": True,
    }
    e.update(extra)
    return e


class TestNormalizeName:
    def test_klein_alias(self):
        assert _normalize_name("#^2 RP^2") == "Klein bottle"

    def test_passthrough(self):
        assert _normalize_name("T^2") == "T^2"


class TestAugmentTriangulation:
    def test_2d_returns_new_entry(self):
        entry = sphere_entry()
        out = _augment_triangulation(
            entry, dimension=2, n_moves=3, rng=random.Random(0)
        )
        assert isinstance(out["triangulation"], list)
        assert out["n_vertices"] == len(
            {v for s in out["triangulation"] for v in s}
        )

    def test_does_not_mutate_input(self):
        entry = sphere_entry()
        before = [list(s) for s in entry["triangulation"]]
        _augment_triangulation(
            entry, dimension=2, n_moves=3, rng=random.Random(0)
        )
        assert entry["triangulation"] == before

    def test_3d_path(self):
        entry = {
            "id": "t0",
            "name": "S^3",
            "n_vertices": 5,
            "triangulation": [list(s) for s in TET_PAIR],
        }
        out = _augment_triangulation(
            entry, dimension=3, n_moves=3, rng=random.Random(0)
        )
        assert isinstance(out["triangulation"], list)


class TestAugmentWithTopologyChange:
    def test_torus_glue_with_genus(self):
        entry = sphere_entry(name="S^2", genus=0)
        out = _augment_with_topology_change(entry, "T^2", rng=random.Random(0))
        assert out["name"] == "T^2"
        assert out["genus"] == 1
        assert out["betti_numbers"] == [1, 2, 1]

    def test_torus_glue_without_genus_key(self):
        entry = sphere_entry(name="S^2")
        entry.pop("genus", None)
        out = _augment_with_topology_change(entry, "T^2", rng=random.Random(0))
        assert "genus" not in out

    def test_torus_glue_nonorientable_genus(self):
        # Gluing a torus to a non-orientable surface adds two
        # crosscaps: Klein bottle (genus 2) -> #^4 RP^2 (genus 4).
        entry = sphere_entry(name="Klein bottle", orientable=False, genus=2)
        out = _augment_with_topology_change(
            entry, "#^4 RP^2", rng=random.Random(0)
        )
        assert out["genus"] == 4

    def test_crosscap_glue(self):
        entry = sphere_entry(name="S^2")
        out = _augment_with_topology_change(
            entry, "RP^2", rng=random.Random(0)
        )
        assert out["name"] == "RP^2"
        assert out["orientable"] is False
        assert out["betti_numbers"] == [1, 0, 0]

    def test_crosscap_glue_genus_from_orientable(self):
        # T^2 (orientable genus 1) + crosscap -> #^3 RP^2, which has
        # non-orientable genus 3 (= 2g + 1 crosscaps).
        entry = sphere_entry(name="T^2", betti_numbers=[1, 2, 1], genus=1)
        out = _augment_with_topology_change(
            entry, "#^3 RP^2", rng=random.Random(0)
        )
        assert out["genus"] == 3

    def test_crosscap_glue_genus_nonorientable(self):
        # RP^2 (genus 1) + crosscap -> Klein bottle (genus 2).
        entry = sphere_entry(name="RP^2", orientable=False, genus=1)
        out = _augment_with_topology_change(
            entry, "Klein bottle", rng=random.Random(0)
        )
        assert out["genus"] == 2

    def test_crosscap_glue_without_genus_key(self):
        entry = sphere_entry(name="S^2")
        entry.pop("genus", None)
        out = _augment_with_topology_change(
            entry, "RP^2", rng=random.Random(0)
        )
        assert "genus" not in out

    def test_normalizes_source_name(self):
        # "#^2 RP^2" normalizes to "Klein bottle"; gluing a torus then
        # targets "#^4 RP^2".
        entry = sphere_entry(name="#^2 RP^2")
        out = _augment_with_topology_change(
            entry, "#^4 RP^2", rng=random.Random(0)
        )
        assert out["name"] == "#^4 RP^2"

    def test_no_matching_target_returns_none(self):
        entry = sphere_entry(name="S^2")
        assert (
            _augment_with_topology_change(
                entry, "#^8 RP^2", rng=random.Random(0)
            )
            is None
        )


class TestFindTopologySources:
    def test_torus_and_empty_and_nonmatching(self):
        class_entries = {
            "S^2": [sphere_entry(name="S^2")],
            "RP^2": [sphere_entry(name="RP^2")],
            "empty": [],
        }
        # S^2 + torus -> T^2; RP^2 maps elsewhere; empty is skipped.
        assert _find_topology_sources("T^2", class_entries) == ["S^2"]

    def test_crosscap_source(self):
        class_entries = {"S^2": [sphere_entry(name="S^2")]}
        # S^2 + crosscap -> RP^2.
        assert _find_topology_sources("RP^2", class_entries) == ["S^2"]


class TestDownsample:
    def test_returns_all_when_below_target(self):
        entries = [sphere_entry(f"s{i}") for i in range(3)]
        assert _downsample(entries, 10, random.Random(0)) is entries

    def test_overshoot_is_trimmed(self):
        # Three vertex strata each contribute one -> 3 > target 2.
        entries = [
            sphere_entry(f"s{i}", nv=nv)
            for i, nv in enumerate([10, 10, 11, 11, 12, 12])
        ]
        out = _downsample(entries, 2, random.Random(0))
        assert len(out) == 2

    def test_lands_exactly_on_target(self):
        # A single stratum of four with target two: proportional
        # allocation hits the target exactly, so neither the trim nor
        # the top-up branch runs.
        entries = [sphere_entry(f"s{i}", nv=10) for i in range(4)]
        out = _downsample(entries, 2, random.Random(0))
        assert len(out) == 2

    def test_undershoot_is_topped_up(self):
        # Two strata of seven; banker's rounding allots four each (8),
        # one short of target 9, so one extra is drawn from the rest.
        entries = [sphere_entry(f"a{i}", nv=10) for i in range(7)] + [
            sphere_entry(f"b{i}", nv=11) for i in range(7)
        ]
        out = _downsample(entries, 9, random.Random(0))
        assert len(out) == 9


class TestBalanceDatasetCore:
    def test_oversamples_small_class(self):
        data = [sphere_entry("s0")]
        out = balance_dataset(
            data,
            dimension=2,
            target_count=3,
            n_moves=2,
            seed=0,
            use_topology_changes=False,
            dedup_max_rounds=0,
        )
        assert len(out) == 3
        assert sum("_aug_" in e["id"] for e in out) == 2

    def test_downsamples_large_class(self):
        data = [sphere_entry(f"s{i}", nv=4 + (i % 3)) for i in range(6)]
        out = balance_dataset(
            data,
            dimension=2,
            target_count=2,
            seed=0,
            use_topology_changes=False,
            dedup_max_rounds=0,
        )
        assert len(out) == 2

    def test_topology_changes_fill_reachable_classes(self):
        data = [sphere_entry(f"s{i}") for i in range(2)]
        out = balance_dataset(
            data,
            dimension=2,
            target_count=2,
            n_moves=2,
            seed=0,
            use_topology_changes=True,
            dedup_max_rounds=0,
        )
        names = {e["name"] for e in out}
        assert "T^2" in names
        assert any("_topo_" in e["id"] for e in out)

    def test_3d_ignores_topology_changes(self):
        data = [
            {
                "id": "t0",
                "name": "S^3",
                "n_vertices": 5,
                "triangulation": [list(s) for s in TET_PAIR],
            }
        ]
        out = balance_dataset(
            data,
            dimension=3,
            target_count=2,
            n_moves=2,
            seed=0,
            use_topology_changes=True,
            dedup_max_rounds=0,
        )
        assert len(out) == 2


class TestBalanceDatasetDedup:
    def test_removes_duplicate_then_regenerates_then_stops(
        self, monkeypatch, capsys
    ):
        calls = {"n": 0}

        def fake(result, verbose=False):
            calls["n"] += 1
            if calls["n"] == 1:
                ids = [e["id"] for e in result]
                return [(ids[0], ids[1])]
            return []

        monkeypatch.setattr(balancing, "find_duplicates", fake)
        data = [sphere_entry("s0")]
        out = balance_dataset(
            data,
            dimension=2,
            target_count=3,
            n_moves=2,
            seed=0,
            use_topology_changes=False,
            dedup_max_rounds=3,
            verbose=True,
        )
        assert len(out) == 3
        # First round removed one and regenerated; second round clean.
        assert calls["n"] == 2
        assert "no duplicates" in capsys.readouterr().err

    def test_last_round_removes_without_regenerating(
        self, monkeypatch, capsys
    ):
        def fake(result, verbose=False):
            ids = [e["id"] for e in result]
            return [(ids[0], ids[1])]

        monkeypatch.setattr(balancing, "find_duplicates", fake)
        data = [sphere_entry("s0")]
        out = balance_dataset(
            data,
            dimension=2,
            target_count=3,
            n_moves=2,
            seed=0,
            use_topology_changes=False,
            dedup_max_rounds=1,
            verbose=True,
        )
        # Single (last) round only removes -> below target, no regen.
        assert len(out) == 2
        assert "removing only" in capsys.readouterr().err

    def test_over_limit_entries_removed(self, monkeypatch, capsys):
        # No isomorphic duplicates; removal is driven purely by the
        # vertex limit (augmented copies grow past it). verbose=True
        # exercises the over-limit branch of the progress message.
        monkeypatch.setattr(
            balancing, "find_duplicates", lambda result, verbose=False: []
        )
        data = [sphere_entry("s0", nv=4)]
        out = balance_dataset(
            data,
            dimension=2,
            target_count=2,
            n_moves=12,
            seed=0,
            use_topology_changes=False,
            dedup_max_rounds=2,
            max_vertices=4,
            verbose=True,
        )
        # The original (within the limit) always survives.
        assert all(e["n_vertices"] <= 4 for e in out if e["id"] == "s0")
        assert "exceeding max_vertices" in capsys.readouterr().err

    def test_regeneration_skips_classes_without_originals(self, monkeypatch):
        # Flag a topology-generated entry (a class with no originals in
        # the dataset) as a duplicate on the first round only.
        state = {"done": False}

        def fake(result, verbose=False):
            if not state["done"]:
                for e in result:
                    if "_topo_" in e["id"]:
                        state["done"] = True
                        return [(e["id"], e["id"])]
            return []

        monkeypatch.setattr(balancing, "find_duplicates", fake)
        data = [sphere_entry(f"s{i}") for i in range(2)]
        out = balance_dataset(
            data,
            dimension=2,
            target_count=2,
            n_moves=2,
            seed=0,
            use_topology_changes=True,
            dedup_max_rounds=3,
        )
        assert isinstance(out, list)


class TestBalanceDatasetMaxVertices:
    def test_prefilter_and_safety_net_when_no_dedup(self):
        # dedup_max_rounds=0 skips the loop; the safety net still
        # enforces the vertex limit on the result.
        data = [
            sphere_entry("keep", nv=4),
            sphere_entry("drop", nv=99),
        ]
        out = balance_dataset(
            data,
            dimension=2,
            target_count=1,
            seed=0,
            use_topology_changes=False,
            dedup_max_rounds=0,
            max_vertices=10,
        )
        assert all(e["n_vertices"] <= 10 for e in out)
        assert "drop" not in {e["id"] for e in out}


class TestPrintStatistics:
    def test_reports_total(self, capsys):
        print_statistics([sphere_entry("s0"), sphere_entry("s1")])
        assert "Total: 2" in capsys.readouterr().out
