"""Tests for the subdivided-dataset generator."""

import json

import pytest

from mantra.generate_subdivided_datasets import (
    _level_tag,
    _resolve_levels,
    _subdivide_entry,
    filter_by_class_count,
    generate_levels,
    main,
    print_statistics,
    subdivide_once,
)


def _entry(triangulation, n_vertices, name="S^2"):
    return {
        "id": "m0",
        "triangulation": triangulation,
        "n_vertices": n_vertices,
        "name": name,
    }


def _triangle(name="S^2"):
    return _entry([[1, 2, 3]], 3, name=name)


class TestHelpers:
    def test_level_tag(self):
        assert _level_tag(1) == "1"
        assert _level_tag(0.5) == "0.5"

    def test_filter_by_class_count_noop(self):
        entries = [_triangle("A"), _triangle("B")]
        filtered, counts = filter_by_class_count(entries, "name", 0)
        assert filtered == entries
        assert counts == {}

    def test_filter_by_class_count_drops(self):
        entries = [_triangle("A"), _triangle("A"), _triangle("B")]
        filtered, counts = filter_by_class_count(entries, "name", 1)
        # "A" occurs twice (> 1) so kept; "B" occurs once (not > 1) dropped
        assert {e["name"] for e in filtered} == {"A"}
        assert counts["A"] == 2

    def test_subdivide_entry_full_barycentric(self):
        out = _subdivide_entry(_triangle(), "full_barycentric")
        assert out["n_vertices"] == 7
        assert out["name"] == "S^2"

    def test_subdivide_entry_stellar(self):
        out = _subdivide_entry(_triangle(), "stellar", fraction=1.0)
        assert out["n_vertices"] == 4

    def test_subdivide_entry_graded(self):
        out = _subdivide_entry(_triangle(), "graded", fraction=5)
        assert out["n_vertices"] == 5

    def test_subdivide_once(self):
        out = subdivide_once([_triangle(), _triangle()], "full_barycentric")
        assert len(out) == 2
        assert all(e["n_vertices"] == 7 for e in out)

    def test_print_statistics(self, capsys):
        print_statistics([_triangle("A"), _triangle("A")])
        captured = capsys.readouterr().out
        assert "Total: 2" in captured


class TestResolveLevels:
    def test_integer(self):
        fractional, n_full, fraction = _resolve_levels("full_barycentric", 3)
        assert (fractional, n_full, fraction) == (False, 3, 1.0)

    def test_fractional_stellar(self):
        fractional, n_full, fraction = _resolve_levels("stellar", 0.5)
        assert fractional is True
        assert n_full == 0
        assert fraction == 0.5

    def test_graded(self):
        fractional, n_full, fraction = _resolve_levels("graded", 12)
        assert fractional is False
        assert n_full == 0
        assert fraction == 12

    def test_non_positive_raises(self):
        with pytest.raises(ValueError):
            _resolve_levels("full_barycentric", 0)

    def test_fractional_non_stellar_raises(self):
        with pytest.raises(ValueError):
            _resolve_levels("full_barycentric", 0.5)

    def test_non_integer_level_raises(self):
        with pytest.raises(ValueError):
            _resolve_levels("full_barycentric", 1.5)

    def test_graded_non_integer_target_raises(self):
        # Non-integer graded targets would hang graded_subdivision; reject.
        with pytest.raises(ValueError):
            _resolve_levels("graded", 5.5)

    def test_graded_non_positive_target_raises(self):
        with pytest.raises(ValueError):
            _resolve_levels("graded", 0)

    def test_graded_returns_integer_target(self):
        fractional, n_full, fraction = _resolve_levels("graded", 12.0)
        assert (fractional, n_full, fraction) == (False, 0, 12)
        assert isinstance(fraction, int)


class TestGenerateLevels:
    def test_unknown_mode(self):
        with pytest.raises(ValueError):
            generate_levels([_triangle()], mode="bogus")

    def test_full_barycentric_multilevel(self):
        out = generate_levels(
            [_triangle()], mode="full_barycentric", n_levels=2
        )
        assert set(out) == {"bary_1", "bary_2"}
        assert out["bary_1"][0]["n_vertices"] == 7
        assert len(out["bary_1"][0]["triangulation"]) == 6
        # second level subdivides the 6-triangle level-1 result in place
        assert len(out["bary_2"]) == 1
        assert len(out["bary_2"][0]["triangulation"]) == 36

    def test_min_vertices_filter(self):
        # level-1 result has 7 vertices; min_vertices=8 drops it, but the
        # (unfiltered) running triangulation still feeds level 2.
        out = generate_levels(
            [_triangle()],
            mode="full_barycentric",
            n_levels=2,
            min_vertices=8,
        )
        assert out["bary_1"] == []
        assert len(out["bary_2"]) == 1
        assert out["bary_2"][0]["n_vertices"] == 25

    def test_fractional_stellar(self):
        out = generate_levels(
            [_entry([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]], 4)],
            mode="stellar",
            n_levels=0.5,
            seed=0,
        )
        assert set(out) == {"bary_0.5"}

    def test_min_class_count(self):
        data = [_triangle("A"), _triangle("A"), _triangle("B")]
        out = generate_levels(
            data,
            mode="full_barycentric",
            n_levels=1,
            min_class_count=1,
        )
        # "B" filtered out before subdivision
        assert {e["name"] for e in out["bary_1"]} == {"A"}

    def test_n_smallest_cohort_multilevel(self):
        data = [_triangle(), _triangle(), _triangle()]
        out = generate_levels(
            data,
            mode="full_barycentric",
            n_levels=2,
            n_smallest=2,
        )
        assert len(out["bary_1"]) == 2
        assert set(out) == {"bary_1", "bary_2"}

    def test_n_smallest_pool_exhausted_warns(self, capsys):
        out = generate_levels(
            [_triangle()],
            mode="full_barycentric",
            n_levels=1,
            n_smallest=5,
        )
        captured = capsys.readouterr().out
        assert "pool exhausted" in captured
        assert len(out["bary_1"]) == 1

    def test_graded_with_n_smallest(self):
        out = generate_levels(
            [_triangle()],
            mode="graded",
            n_levels=5,
            n_smallest=1,
        )
        assert set(out) == {"graded_n5"}
        assert out["graded_n5"][0]["n_vertices"] == 5

    def test_graded_without_n_smallest_is_empty(self):
        out = generate_levels([_triangle()], mode="graded", n_levels=5)
        assert out == {}

    def test_graded_skips_unchanged_entries(self, capsys):
        # An entry already at/above the target is a graded no-op; it must be
        # skipped so the un-subdivided (raw) triangulation does not leak into
        # the output and duplicate the original sample in training.
        big = _entry([[1, 2, 3], [4, 5, 6]], 6, name="A")
        out = generate_levels([big], mode="graded", n_levels=5, n_smallest=1)
        assert out["graded_n5"] == []
        assert "pool exhausted" in capsys.readouterr().out

    def test_graded_skips_unchanged_keeps_grown(self):
        # Within one class: the small entry grows to the target and is kept;
        # the already-large entry is a no-op and is skipped.
        data = [
            _entry([[1, 2, 3], [4, 5, 6]], 6, name="A"),
            _triangle("A"),
        ]
        out = generate_levels(data, mode="graded", n_levels=5, n_smallest=2)
        assert len(out["graded_n5"]) == 1
        assert out["graded_n5"][0]["n_vertices"] == 5

    def test_n_smallest_fractional_stellar(self):
        out = generate_levels(
            [_entry([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]], 4)],
            mode="stellar",
            n_levels=0.5,
            n_smallest=1,
            seed=0,
        )
        assert set(out) == {"bary_0.5"}

    def test_n_smallest_drops_below_min_vertices(self, capsys):
        # A triangle subdivides to 7 vertices, below min_vertices=8, so it is
        # dropped and the per-class pool is exhausted with nothing kept.
        out = generate_levels(
            [_triangle()],
            mode="full_barycentric",
            n_levels=1,
            min_vertices=8,
            n_smallest=1,
        )
        assert out["bary_1"] == []
        assert "pool exhausted" in capsys.readouterr().out


class TestMain:
    def test_end_to_end(self, tmp_path, capsys):
        data = [_triangle(), _triangle("T^2")]
        input_path = tmp_path / "in.json"
        input_path.write_text(json.dumps(data))
        out_dir = tmp_path / "out"

        main(
            [
                "--input",
                str(input_path),
                "--output-dir",
                str(out_dir),
                "--prefix",
                "2_manifolds",
                "--mode",
                "full_barycentric",
                "--n-levels",
                "1",
            ]
        )

        out_file = out_dir / "2_manifolds_bary_1.json"
        assert out_file.exists()
        written = json.loads(out_file.read_text())
        assert len(written) == 2
        assert all(e["n_vertices"] == 7 for e in written)
        assert "[INFO] Loaded 2" in capsys.readouterr().out

    def test_empty_level_output(self, tmp_path):
        # min_vertices above the subdivided count yields an empty level,
        # exercising the no-stats branch in main().
        data = [_triangle()]
        input_path = tmp_path / "in.json"
        input_path.write_text(json.dumps(data))
        out_dir = tmp_path / "out"

        main(
            [
                "--input",
                str(input_path),
                "--output-dir",
                str(out_dir),
                "--prefix",
                "2_manifolds",
                "--mode",
                "full_barycentric",
                "--n-levels",
                "1",
                "--min-vertices",
                "100",
            ]
        )

        written = json.loads((out_dir / "2_manifolds_bary_1.json").read_text())
        assert written == []
