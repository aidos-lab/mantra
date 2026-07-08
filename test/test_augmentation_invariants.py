"""Topological invariant tests for Pachner moves and gluing.

Rather than checking implementation details, these tests compute
invariants (Euler characteristic, f-vector, orientability, manifold
validity) from the resulting triangulations and verify they behave as
the underlying topology dictates: Pachner moves preserve the homeo-
morphism type, gluing a torus drops chi by 2, gluing a crosscap drops
chi by 1 and kills orientability, and so on.
"""

import random
from collections import defaultdict
from itertools import combinations

from mantra.augmentations.balancing import (
    _augment_triangulation,
    _augment_with_topology_change,
)
from mantra.augmentations.constants import (
    RP2_TRIANGULATION_MINUS_FACE,
    TORUS_TRIANGULATION_MINUS_FACE,
)
from mantra.augmentations.triangulation import Triangulation, Triangulation2D

# Boundary of a tetrahedron: the minimal 2-sphere (chi = 2).
SPHERE = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
# Completing the surface-minus-face pieces with the missing triangle
# {1, 2, 3} yields closed surfaces: the 7-vertex torus (chi = 0) and
# the 6-vertex real projective plane (chi = 1).
TORUS = TORUS_TRIANGULATION_MINUS_FACE + [[1, 2, 3]]
RP2 = RP2_TRIANGULATION_MINUS_FACE + [[1, 2, 3]]
# Boundary of the 4-simplex: the minimal 3-sphere (chi = 0).
S3 = [list(c) for c in combinations(range(1, 6), 4)]


def is_orientable(triangulation):
    """Check orientability of a closed connected surface.

    Propagates triangle orientations across shared edges: a surface is
    orientable iff every triangle can be assigned a cyclic vertex order
    such that adjacent triangles traverse their shared edge in opposite
    directions.
    """
    tris = [tuple(sorted(t)) for t in triangulation]
    edge_to_tris = defaultdict(list)
    for idx, (a, b, c) in enumerate(tris):
        for u, v in ((a, b), (b, c), (a, c)):
            edge_to_tris[(u, v)].append(idx)

    def traverses_forward(idx, sign, u, v):
        a, b, c = tris[idx]
        cycle = (a, b, c) if sign == 1 else (a, c, b)
        directed = {
            (cycle[0], cycle[1]),
            (cycle[1], cycle[2]),
            (cycle[2], cycle[0]),
        }
        return (u, v) in directed

    signs = {0: 1}
    stack = [0]
    while stack:
        i = stack.pop()
        a, b, c = tris[i]
        for u, v in ((a, b), (b, c), (a, c)):
            forward = traverses_forward(i, signs[i], u, v)
            for j in edge_to_tris[(u, v)]:
                if j == i:
                    continue
                needed = -1 if traverses_forward(j, 1, u, v) == forward else 1
                if j in signs:
                    if signs[j] != needed:
                        return False
                else:
                    signs[j] = needed
                    stack.append(j)
    return True


class TestOrientabilityChecker:
    """Sanity-check the helper against surfaces of known type."""

    def test_sphere_is_orientable(self):
        assert is_orientable(SPHERE) is True

    def test_torus_is_orientable(self):
        assert is_orientable(TORUS) is True

    def test_rp2_is_not_orientable(self):
        assert is_orientable(RP2) is False


class Test2DPachnerInvariants:
    def test_subdivide_f_vector_delta(self):
        # A 1-3 move adds one vertex, three edges and one net triangle
        # (one removed, three added); chi is unchanged.
        t = Triangulation.from_list(TORUS)
        f0, f1, f2 = t.f_vector()
        t.subdivide(frozenset({1, 2, 4}))
        assert t.f_vector() == (f0 + 1, f1 + 3, f2 + 2)
        assert t.euler_characteristic() == 0
        t.validate()

    def test_flip_preserves_f_vector(self):
        # The minimal torus has a complete 1-skeleton, so subdivide
        # once to create a flippable edge. A 2-2 move then leaves the
        # whole f-vector unchanged.
        t = Triangulation.from_list(TORUS, rng=random.Random(42))
        t.subdivide(frozenset({1, 2, 4}))
        f_before = t.f_vector()
        assert t.flip_edge() is True
        assert t.f_vector() == f_before
        assert t.euler_characteristic() == 0
        t.validate()

    def test_flip_is_an_involution(self):
        # Flipping {1, 2} creates {3, 8}; flipping that edge restores
        # the original triangulation exactly.
        t = Triangulation.from_list(TORUS)
        t.subdivide(frozenset({1, 2, 4}))
        before = set(t._simplices)
        assert t.flip_edge(frozenset({1, 2})) is True
        assert t.flip_edge(frozenset({3, 8})) is True
        assert t._simplices == before

    def test_random_moves_preserve_torus_invariants(self):
        t = Triangulation.from_list(TORUS, rng=random.Random(42))
        for _ in range(15):
            t.random_pachner_move()
            assert t.euler_characteristic() == 0
            t.validate()
        assert is_orientable(t.to_list()) is True

    def test_random_moves_preserve_rp2_invariants(self):
        t = Triangulation.from_list(RP2, rng=random.Random(42))
        for _ in range(15):
            t.random_pachner_move()
            assert t.euler_characteristic() == 1
            t.validate()
        assert is_orientable(t.to_list()) is False


class TestGlueInvariants:
    def test_glue_torus_drops_chi_by_two(self):
        t = Triangulation.from_list(SPHERE, rng=random.Random(42))
        assert t.euler_characteristic() == 2
        t.glue("torus")
        assert t.euler_characteristic() == 0
        assert is_orientable(t.to_list()) is True
        t.validate()

    def test_glue_crosscap_drops_chi_by_one_and_orientability(self):
        t = Triangulation.from_list(SPHERE, rng=random.Random(42))
        t.glue("crosscap")
        assert t.euler_characteristic() == 1
        assert is_orientable(t.to_list()) is False
        t.validate()

    def test_torus_and_klein_bottle_differ_only_in_orientability(self):
        # S^2 + torus and S^2 + 2 crosscaps both have chi = 0, but
        # only the former is orientable.
        t1 = Triangulation.from_list(SPHERE, rng=random.Random(42))
        t1.glue("torus")
        t2 = Triangulation.from_list(SPHERE, rng=random.Random(42))
        t2.glue("crosscap")
        t2.glue("crosscap")
        assert t1.euler_characteristic() == 0
        assert t2.euler_characteristic() == 0
        assert is_orientable(t1.to_list()) is True
        assert is_orientable(t2.to_list()) is False
        t2.validate()

    def test_genus_chi_relation_orientable(self):
        # chi(orientable genus g) = 2 - 2g.
        for g in (1, 2, 3):
            t = Triangulation.from_list(SPHERE, rng=random.Random(42))
            for _ in range(g):
                t.glue("torus")
            assert t.euler_characteristic() == 2 - 2 * g
            assert is_orientable(t.to_list()) is True
            t.validate()

    def test_genus_chi_relation_nonorientable(self):
        # chi(non-orientable genus k) = 2 - k.
        for k in (1, 2, 3):
            t = Triangulation.from_list(SPHERE, rng=random.Random(k))
            for _ in range(k):
                t.glue("crosscap")
            assert t.euler_characteristic() == 2 - k
            assert is_orientable(t.to_list()) is False
            t.validate()

    def test_glue_torus_on_nonorientable_adds_two_crosscaps(self):
        # RP^2 + torus = #^3 RP^2: chi = 1 - 2 = -1 = 2 - 3, and the
        # surface stays non-orientable.
        t = Triangulation.from_list(RP2, rng=random.Random(42))
        t.glue("torus")
        assert t.euler_characteristic() == -1
        assert is_orientable(t.to_list()) is False
        t.validate()

    def test_glue_then_pachner_moves_keep_invariants(self):
        # The balancing pipeline applies Pachner moves on top of a
        # topology change; the homeomorphism type must survive both.
        t = Triangulation.from_list(SPHERE, rng=random.Random(42))
        t.glue("torus")
        for _ in range(10):
            t.random_pachner_move()
        assert t.euler_characteristic() == 0
        assert is_orientable(t.to_list()) is True
        t.validate()


class TestTopologyChangeMetadataConsistency:
    """The metadata written by ``_augment_with_topology_change`` must
    agree with invariants computed from the produced triangulation."""

    @staticmethod
    def assert_consistent(out):
        chi = Triangulation2D(out["triangulation"]).euler_characteristic()
        chi = Triangulation.from_list(
            out["triangulation"]
        ).euler_characteristic()
        b0, b1, b2 = out["betti_numbers"]
        assert chi == b0 - b1 + b2
        orientable = is_orientable(out["triangulation"])
        assert orientable is out["orientable"]
        if orientable:
            assert chi == 2 - 2 * out["genus"]
        else:
            assert chi == 2 - out["genus"]

    def test_sphere_to_torus(self):
        entry = {
            "id": "s0",
            "name": "S^2",
            "n_vertices": 4,
            "triangulation": [list(s) for s in SPHERE],
            "betti_numbers": [1, 0, 1],
            "orientable": True,
            "genus": 0,
        }
        out = _augment_with_topology_change(
            entry, glue_type="torus", rng=random.Random(42)
        )
        self.assert_consistent(out)

    def test_rp2_to_klein_bottle(self):
        entry = {
            "id": "r0",
            "name": "RP^2",
            "n_vertices": 6,
            "triangulation": [list(s) for s in RP2],
            "betti_numbers": [1, 0, 0],
            "orientable": False,
            "genus": 1,
        }
        out = _augment_with_topology_change(
            entry, glue_type="crosscap", rng=random.Random(42)
        )
        self.assert_consistent(out)

    def test_torus_to_3_crosscaps(self):
        entry = {
            "id": "t0",
            "name": "T^2",
            "n_vertices": 7,
            "triangulation": [list(s) for s in TORUS],
            "betti_numbers": [1, 2, 1],
            "orientable": True,
            "genus": 1,
        }
        out = _augment_with_topology_change(
            entry, glue_type="torus", rng=random.Random(42)
        )
        self.assert_consistent(out)

    def test_klein_bottle_to_4_crosscaps(self):
        builder = Triangulation.from_list(RP2, rng=random.Random(42))
        builder.glue("crosscap")
        klein = builder.to_list()
        entry = {
            "id": "k0",
            "name": "Klein bottle",
            "n_vertices": builder.n_vertices,
            "triangulation": klein,
            "betti_numbers": [1, 1, 0],
            "orientable": False,
            "genus": 2,
        }
        out = _augment_with_topology_change(
            entry, glue_type="crosscap", rng=random.Random(42)
        )
        self.assert_consistent(out)


class Test3DPachnerInvariants:
    def test_move_f_vector_deltas(self):
        t = Triangulation.from_list(S3, rng=random.Random(42))
        assert t.f_vector() == (5, 10, 10, 5)
        assert t.euler_characteristic() == 0

        # 1-4: one new vertex, joined to the 4 faces of the old tet.
        assert t.move_1_4(frozenset({1, 2, 3, 4})) is True
        assert t.f_vector() == (6, 14, 16, 8)
        assert t.euler_characteristic() == 0
        t.validate()

        # 2-3: one new edge, two new faces, one net tetrahedron.
        assert t.move_2_3(frozenset({1, 2, 3})) is True
        assert t.f_vector() == (6, 15, 18, 9)
        assert t.euler_characteristic() == 0
        t.validate()

    def test_inverse_moves_restore_the_sphere(self):
        # 1-4 / 2-3 followed by their inverses 3-2 / 4-1 give back the
        # boundary of the 4-simplex exactly.
        t = Triangulation.from_list(S3, rng=random.Random(42))
        original = set(t._simplices)

        assert t.move_1_4(frozenset({1, 2, 3, 4})) is True
        after_1_4 = set(t._simplices)
        # Face {1, 2, 3} sits between tets {1,2,3,5} and {1,2,3,6};
        # the opposite edge {5, 6} does not exist yet.
        assert t.move_2_3(frozenset({1, 2, 3})) is True
        assert t.move_3_2(frozenset({5, 6})) is True
        assert set(t._simplices) == after_1_4
        assert t.move_4_1(6) is True
        assert set(t._simplices) == original

    def test_random_walk_preserves_invariants(self):
        t = Triangulation.from_list(S3, rng=random.Random(42))
        for _ in range(20):
            assert t.random_pachner_move() is True
            assert t.euler_characteristic() == 0
            t.validate()


def assert_canonical_labels(tri_list, expected_n_vertices):
    """Assert the exported labeling contract of ``to_list``.

    Vertex labels must be exactly 1..n with no gaps, every simplex
    must be sorted, and the simplex list itself must be sorted.
    """
    verts = {v for s in tri_list for v in s}
    assert verts == set(range(1, len(verts) + 1))
    assert len(verts) == expected_n_vertices
    assert all(s == sorted(s) for s in tri_list)
    assert tri_list == sorted(tri_list)


class TestVertexLabeling:
    def test_to_list_canonical_after_random_2d_moves(self):
        t = Triangulation.from_list(TORUS, rng=random.Random(42))
        for _ in range(15):
            t.random_pachner_move()
        assert_canonical_labels(t.to_list(), t.n_vertices)

    def test_to_list_compacts_gap_left_by_4_1_move(self):
        # Subdivide twice (vertices 6 and 7), then collapse vertex 6:
        # the internal label space now has a gap at 6, which to_list
        # must compact back to 1..6 without altering the complex.
        t = Triangulation.from_list(S3, rng=random.Random(42))
        assert t.move_1_4(frozenset({1, 2, 3, 4})) is True
        assert t.move_1_4(frozenset({1, 2, 3, 5})) is True
        assert t.move_4_1(6) is True
        assert t.vertices == {1, 2, 3, 4, 5, 7}
        exported = t.to_list()
        assert_canonical_labels(exported, 6)
        # relabeling must not change the combinatorial structure
        assert Triangulation.from_list(exported).f_vector() == t.f_vector()

    def test_glue_allocates_fresh_vertex_labels(self):
        t = Triangulation.from_list(SPHERE)
        t.glue("torus", frozenset({1, 2, 3}))
        # 4 original vertices plus 4 interior torus vertices, with the
        # new labels allocated above the existing ones (no collision).
        assert t.vertices == set(range(1, 9))
        # the boundary triangle was consumed by the gluing
        assert frozenset({1, 2, 3}) not in t._simplices
        assert_canonical_labels(t.to_list(), 8)

    def test_augmented_entry_labeling_is_consistent(self):
        entry = {
            "id": "s0",
            "name": "S^2",
            "n_vertices": 4,
            "triangulation": [list(s) for s in SPHERE],
        }
        out = _augment_triangulation(entry, n_moves=8, rng=random.Random(42))
        assert_canonical_labels(out["triangulation"], out["n_vertices"])
