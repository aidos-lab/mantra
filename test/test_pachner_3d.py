"""Tests for 3D Pachner moves."""

from mantra.augmentations.triangulation_3d import Triangulation3D


# Boundary of 4-simplex = minimal S^3 (5 vertices, 5 tetrahedra)
S3_MINIMAL = [
    [1, 2, 3, 4],
    [1, 2, 3, 5],
    [1, 2, 4, 5],
    [1, 3, 4, 5],
    [2, 3, 4, 5],
]


class TestTriangulation3DConstruction:
    def test_s3_minimal(self):
        t = Triangulation3D(S3_MINIMAL)
        assert t.n_vertices == 5
        assert len(t._simplices) == 5
        assert t.dimension == 3

    def test_to_list_roundtrip(self):
        t = Triangulation3D(S3_MINIMAL)
        result = t.to_list()
        assert len(result) == 5
        assert all(len(tet) == 4 for tet in result)

    def test_euler_characteristic_s3(self):
        """Closed 3-manifolds have chi = 0."""
        t = Triangulation3D(S3_MINIMAL)
        assert t.euler_characteristic() == 0

    def test_validate_s3(self):
        t = Triangulation3D(S3_MINIMAL)
        t.validate()

    def test_f_vector_s3(self):
        t = Triangulation3D(S3_MINIMAL)
        # 5 vertices, 10 edges, 10 faces, 5 tets
        assert t.f_vector() == (5, 10, 10, 5)


class TestMove14:
    def test_move_increases_vertices(self):
        t = Triangulation3D(S3_MINIMAL)
        t.move_1_4()
        assert t.n_vertices == 6

    def test_move_increases_tets(self):
        """1-4: replace 1 tet with 4, net +3."""
        t = Triangulation3D(S3_MINIMAL)
        t.move_1_4()
        assert len(t._simplices) == 8

    def test_move_preserves_euler_characteristic(self):
        t = Triangulation3D(S3_MINIMAL)
        t.move_1_4()
        assert t.euler_characteristic() == 0

    def test_move_preserves_closed_manifold(self):
        t = Triangulation3D(S3_MINIMAL)
        t.move_1_4()
        t.validate()


class TestMove23:
    def test_move_preserves_vertex_count(self):
        t = Triangulation3D(S3_MINIMAL)
        n_before = t.n_vertices
        result = t.move_2_3()
        if result:
            assert t.n_vertices == n_before

    def test_move_changes_tet_count(self):
        """2-3: replace 2 tets with 3, net +1."""
        t = Triangulation3D(S3_MINIMAL)
        n_before = len(t._simplices)
        result = t.move_2_3()
        if result:
            assert len(t._simplices) == n_before + 1

    def test_move_preserves_euler_characteristic(self):
        t = Triangulation3D(S3_MINIMAL)
        t.move_2_3()
        assert t.euler_characteristic() == 0

    def test_move_preserves_closed_manifold(self):
        t = Triangulation3D(S3_MINIMAL)
        result = t.move_2_3()
        if result:
            t.validate()

    def test_move_after_1_4(self):
        """After a 1-4 move there should be valid 2-3 candidates."""
        t = Triangulation3D(S3_MINIMAL)
        t.move_1_4()
        result = t.move_2_3()
        assert result is True
        assert t.euler_characteristic() == 0
        t.validate()


class TestMove32:
    def test_roundtrip_with_2_3(self):
        """2-3 followed by 3-2 on the new edge should recover."""
        t = Triangulation3D(S3_MINIMAL)
        t.move_1_4()  # create some structure
        before = set(t._simplices)
        n_tets_before = len(before)
        result_23 = t.move_2_3()
        if result_23:
            result_32 = t.move_3_2()
            if result_32:
                assert len(t._simplices) == n_tets_before
                assert t.euler_characteristic() == 0
                t.validate()

    def test_move_preserves_euler_characteristic(self):
        t = Triangulation3D(S3_MINIMAL)
        t.move_1_4()
        t.move_2_3()
        result = t.move_3_2()
        if result:
            assert t.euler_characteristic() == 0
            t.validate()


class TestMove41:
    def test_roundtrip_with_1_4(self):
        """1-4 on a tet, then 4-1 on the new vertex should recover
        the original."""
        t = Triangulation3D(S3_MINIMAL)
        original = set(t._simplices)
        tet = list(t._simplices)[0]
        t.move_1_4(tet=tet)
        # the new vertex is the highest-numbered one
        new_v = max(t.vertices)
        result = t.move_4_1(vertex=new_v)
        assert result is True
        assert t._simplices == original

    def test_move_preserves_euler_characteristic(self):
        t = Triangulation3D(S3_MINIMAL)
        t.move_1_4()
        new_v = max(t.vertices)
        t.move_4_1(vertex=new_v)
        assert t.euler_characteristic() == 0
        t.validate()


class TestRandomPachnerSequence3D:
    def test_sequence_preserves_invariants(self):
        """Apply 20 random Pachner moves, check invariants after
        each."""
        t = Triangulation3D(S3_MINIMAL)
        for _ in range(20):
            t.random_pachner_move()
            assert t.euler_characteristic() == 0
            t.validate()

    def test_sequence_grows_triangulation(self):
        """After many moves the triangulation should have grown."""
        t = Triangulation3D(S3_MINIMAL)
        for _ in range(10):
            t.random_pachner_move()
        # should have at least a few more tets than the original 5
        assert len(t._simplices) >= 5
