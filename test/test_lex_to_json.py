"""Tests for lex_to_json parsing functions."""

import pytest

from mantra.lex_to_json import (
    process_triangulation,
    parse_topological_type,
    parse_homology_groups,
    hash_triangulation,
    find_duplicates,
    guess_name,
)


class TestProcessTriangulation:
    def test_single_triangle(self):
        result = process_triangulation("1,2,3")
        assert result["triangulation"] == [[1, 2, 3]]
        assert result["dimension"] == 2
        assert result["n_vertices"] == 3

    def test_two_triangles(self):
        result = process_triangulation("1,2,3\n1,2,4")
        assert len(result["triangulation"]) == 2
        assert result["dimension"] == 2
        assert result["n_vertices"] == 4

    def test_tetrahedron(self):
        result = process_triangulation("1,2,3,4")
        assert result["dimension"] == 3
        assert result["n_vertices"] == 4


class TestParseTopologicalType:
    def test_orientable(self):
        result = parse_topological_type("(+; 1) = T^2")
        assert result["orientable"] is True
        assert result["genus"] == 1
        assert result["name"] == "T^2"

    def test_non_orientable(self):
        result = parse_topological_type("(-; 1) = RP^2")
        assert result["orientable"] is False
        assert result["genus"] == 1
        assert result["name"] == "RP^2"

    def test_name_only(self):
        result = parse_topological_type("S^2")
        assert result["name"] == "S^2"
        assert "orientable" not in result

    def test_unnamed(self):
        result = parse_topological_type("(+; 3)")
        assert result["orientable"] is True
        assert result["genus"] == 3
        assert result["name"] == ""


class TestParseHomologyGroups:
    def test_sphere(self):
        result = parse_homology_groups("(1, 0, 1)")
        assert result["betti_numbers"] == [1, 0, 1]
        assert result["torsion_coefficients"] == ["", "", ""]

    def test_with_torsion(self):
        result = parse_homology_groups("(1, 0 + Z_2, 0)")
        assert result["betti_numbers"] == [1, 0, 0]
        assert result["torsion_coefficients"] == ["", "Z_2", ""]

    def test_3_manifold(self):
        result = parse_homology_groups("(1, 0, 0, 1)")
        assert result["betti_numbers"] == [1, 0, 0, 1]


class TestHashTriangulation:
    def test_order_invariant(self):
        tri1 = [[1, 2, 3], [1, 2, 4]]
        tri2 = [[1, 2, 4], [1, 2, 3]]
        assert hash_triangulation(tri1) == hash_triangulation(tri2)

    def test_vertex_order_invariant(self):
        tri1 = [[1, 2, 3]]
        tri2 = [[3, 1, 2]]
        assert hash_triangulation(tri1) == hash_triangulation(tri2)

    def test_different_triangulations_differ(self):
        tri1 = [[1, 2, 3]]
        tri2 = [[1, 2, 4]]
        assert hash_triangulation(tri1) != hash_triangulation(tri2)


class TestFindDuplicates:
    def test_no_duplicates(self):
        tris = {
            "a": {"triangulation": [[1, 2, 3]]},
            "b": {"triangulation": [[1, 2, 4]]},
        }
        assert find_duplicates(tris) == []

    def test_finds_duplicate(self):
        tris = {
            "a": {"triangulation": [[1, 2, 3]]},
            "b": {"triangulation": [[3, 1, 2]]},
        }
        dups = find_duplicates(tris)
        assert len(dups) == 1
        assert dups[0] == "b"  # second one is the duplicate


class TestGuessName:
    def test_orientable_genus_2(self):
        tri = {
            "name": "",
            "dimension": 2,
            "orientable": True,
            "genus": 2,
            "betti_numbers": [1, 4, 1],
        }
        name = guess_name(tri)
        assert name == "#^2 T^2"

    def test_non_orientable_genus_2(self):
        tri = {
            "name": "",
            "dimension": 2,
            "orientable": False,
            "genus": 2,
            "betti_numbers": [1, 1, 0],
            "torsion_coefficients": ["", "Z_2", ""],
        }
        name = guess_name(tri)
        assert name == "#^2 RP^2"

    def test_already_named_raises(self):
        tri = {"name": "S^2", "dimension": 2}
        with pytest.raises(AssertionError):
            guess_name(tri)

    def test_3d_returns_empty(self):
        tri = {"name": "", "dimension": 3}
        assert guess_name(tri) == ""
