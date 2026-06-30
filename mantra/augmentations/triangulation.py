"""Base class for mutable triangulation data structures."""

import random
from collections import defaultdict
from itertools import combinations
from abc import ABC, abstractmethod

from constants import _RP2_TRIANGULATION_MINUS_FACE, _TORUS_TRIANGULATION_MINUS_FACE


class Triangulation(ABC):
    """Mutable triangulation stored as a set of frozensets.

    Parameters
    ----------
    top_simplices : list of list of int
        Top-level simplices with 1-indexed vertices.
    dimension : int
        Dimension of the triangulation (2 or 3).
    rng : random.Random or None
        Random number generator. If None, the module-level
        ``random`` is used.
    """

    def __init__(self, top_simplices, dimension, rng=None):
        self._dim = dimension
        self._simplices = {frozenset(s) for s in top_simplices}
        self._next_vertex = max(v for s in self._simplices for v in s) + 1
        self._rng = rng if rng is not None else random

    @staticmethod
    def from_list(triangulation, rng=None) -> 'Triangulation':
        """Construct a triangulation from a list of lists.

        Parameters
        ----------
        triangulation : list of list of int
            Top-level simplices with 1-indexed vertices.
        rng : random.Random or None
            Random number generator. If None, the module-level
            ``random`` is used.

        Returns
        -------
        Triangulation
            New triangulation object.
        """
        dim = len(triangulation[0]) - 1
        if dim == 2:
            return Triangulation2D(triangulation, rng=rng)
        elif dim == 3:
            return Triangulation3D(triangulation, rng=rng)
        else:
            raise ValueError(f"Unsupported dimension: {dim}")
    @property
    def n_vertices(self):
        """Return the number of distinct vertices."""
        return len(self.vertices)

    @property
    def vertices(self):
        """Return the set of all vertex labels."""
        return {v for s in self._simplices for v in s}

    @property
    def dimension(self):
        """Return the dimension of the triangulation."""
        return self._dim

    def _new_vertex(self):
        """Allocate and return a new vertex label."""
        v = self._next_vertex
        self._next_vertex += 1
        return v

    def to_list(self):
        """Export triangulation as sorted list of sorted lists.

        Vertex labels are remapped to a contiguous ``1..n_vertices``
        range. Pachner moves that remove vertices (e.g. the 4-1 move
        in 3D) can leave gaps in the label space, and the allocator
        for new labels never reuses them. Compacting on export gives
        callers the canonical invariant ``max(label) == n_vertices``.
        """
        used = sorted({v for s in self._simplices for v in s})
        remap = {old: new for new, old in enumerate(used, start=1)}
        return sorted(sorted(remap[v] for v in s) for s in self._simplices)

    def face_to_cofaces(self, face_dim):
        """Map faces of given dimension to their containing top-simplices.

        Parameters
        ----------
        face_dim : int
            Dimension of the faces (number of vertices - 1).

        Returns
        -------
        dict[frozenset, list[frozenset]]
            Mapping from each face to the list of top-simplices
            containing it.
        """
        result = defaultdict(list)
        k = face_dim + 1  # number of vertices in a face
        for s in self._simplices:
            for face in combinations(s, k):
                result[frozenset(face)].append(s)
        return result

    def _all_faces(self, dim):
        """Return all faces of a given dimension as a set."""
        k = dim + 1
        faces = set()
        for s in self._simplices:
            for face in combinations(s, k):
                faces.add(frozenset(face))
        return faces

    def f_vector(self):
        """Compute the f-vector (counts of k-simplices for all k).

        Returns
        -------
        tuple of int
            (f_0, f_1, ..., f_d) where f_k is the number of
            k-simplices.
        """
        return tuple(len(self._all_faces(k)) for k in range(self._dim + 1))

    def euler_characteristic(self):
        """Compute the Euler characteristic from the f-vector."""
        fv = self.f_vector()
        return sum((-1) ** k * fv[k] for k in range(len(fv)))

    def validate(self):
        """Check that every codimension-1 face has exactly 2 cofaces.

        This is a necessary condition for a closed manifold
        triangulation.

        Raises
        ------
        ValueError
            If any codimension-1 face does not have exactly 2 cofaces.
        """
        codim1 = self.face_to_cofaces(self._dim - 1)
        for face, cofaces in codim1.items():
            if len(cofaces) != 2:
                raise ValueError(
                    f"Face {set(face)} has {len(cofaces)} cofaces, "
                    f"expected 2 for a closed manifold."
                )

class Triangulation2D(Triangulation):
    """Mutable 2D triangulation supporting Pachner moves.

    Parameters
    ----------
    top_simplices : list of list of int
        Triangles as lists of 3 vertex labels (1-indexed).
    """

    def __init__(self, top_simplices, rng=None):
        super().__init__(top_simplices, dimension=2, rng=rng)

    def flip_edge(self, edge=None):
        """Perform a 2-2 Pachner move (edge flip).

        Parameters
        ----------
        edge : frozenset of int or None
            Edge to flip. If None, a random flippable edge is chosen.

        Returns
        -------
        bool
            True if the flip was performed, False if no valid flip
            exists.
        """
        edge_cofaces = self.face_to_cofaces(1)

        if edge is None:
            # find all flippable edges
            flippable = []
            for e, cofaces in edge_cofaces.items():
                if len(cofaces) != 2:
                    continue
                all_verts = cofaces[0] | cofaces[1]
                new_edge = all_verts - e
                # check new edge doesn't already connect two triangles
                if new_edge not in edge_cofaces:
                    flippable.append(e)
            if not flippable:
                return False
            edge = self._rng.choice(flippable)

        cofaces = edge_cofaces[edge]
        if len(cofaces) != 2:
            return False

        all_verts = cofaces[0] | cofaces[1]
        new_edge = all_verts - edge
        if new_edge in edge_cofaces:
            return False

        # remove old triangles, add new ones
        self._simplices.discard(cofaces[0])
        self._simplices.discard(cofaces[1])

        for v in edge:
            self._simplices.add(frozenset(new_edge | {v}))

        return True

    def subdivide(self, triangle=None):
        """Perform a 1-3 Pachner move (stellar subdivision).

        Parameters
        ----------
        triangle : frozenset of int or None
            Triangle to subdivide. If None, a random triangle is
            chosen.

        Returns
        -------
        bool
            Always True.
        """
        if triangle is None:
            triangle = self._rng.choice(list(self._simplices))

        self._simplices.discard(triangle)
        v = self._new_vertex()

        for edge in combinations(triangle, 2):
            self._simplices.add(frozenset(edge) | {v})

        return True

    def glue_torus(self, triangle=None):
        """Connected sum with a torus (increases genus by 1).

        Removes a triangle and glues in a torus with one triangle
        removed, identifying the boundary.

        Parameters
        ----------
        triangle : frozenset of int or None
            Triangle to remove for gluing. If None, a random triangle
            is chosen.

        Returns
        -------
        dict
            Metadata updates: keys that should be updated on the
            data dict (e.g. betti_numbers, name, orientable).
        """
        if triangle is None:
            triangle = self._rng.choice(list(self._simplices))

        return self._glue_surface(
            triangle,
            _TORUS_TRIANGULATION_MINUS_FACE,
            n_new_vertices=4,
        )

    def glue_crosscap(self, triangle=None):
        """Connected sum with RP^2 (adds one crosscap).

        Removes a triangle and glues in an RP^2 with one triangle
        removed, identifying the boundary.

        Parameters
        ----------
        triangle : frozenset of int or None
            Triangle to remove for gluing. If None, a random triangle
            is chosen.

        Returns
        -------
        dict
            Metadata updates.
        """
        if triangle is None:
            triangle = self._rng.choice(list(self._simplices))

        return self._glue_surface(
            triangle,
            _RP2_TRIANGULATION_MINUS_FACE,
            n_new_vertices=3,
        )

    def _glue_surface(self, triangle, surface_triangles, n_new_vertices):
        """Glue a surface-minus-face to this triangulation.

        The surface must have boundary vertices {1, 2, 3} which get
        identified with the vertices of the removed triangle.

        Parameters
        ----------
        triangle : frozenset of int
            Triangle to remove.
        surface_triangles : list of list of int
            Triangulation of the surface minus one face, with
            boundary vertices labeled 1, 2, 3.
        n_new_vertices : int
            Number of interior vertices in the surface piece.
        """
        self._simplices.discard(triangle)

        # map boundary vertices {1,2,3} to the triangle's vertices,
        # and interior vertices {4, ..., 3+n} to new vertices
        boundary_verts = sorted(triangle)
        assignment = {}
        for i, v in enumerate(boundary_verts):
            assignment[i + 1] = v
        for i in range(n_new_vertices):
            assignment[4 + i] = self._new_vertex()

        for tri in surface_triangles:
            new_tri = frozenset(assignment[v] for v in tri)
            self._simplices.add(new_tri)

        return {}

    def random_pachner_move(self, weights=None):
        """Apply a random Pachner move.

        Parameters
        ----------
        weights : tuple of float or None
            Weights for (flip_edge, subdivide). Default: equal.

        Returns
        -------
        bool
            True if a move was performed.
        """
        if weights is None:
            weights = (1.0, 1.0)

        moves = [self.flip_edge, self.subdivide]
        move = self._rng.choices(moves, weights=weights, k=1)[0]
        return move()

class Triangulation3D(Triangulation):
    """Mutable 3D triangulation supporting Pachner moves.

    Parameters
    ----------
    top_simplices : list of list of int
        Tetrahedra as lists of 4 vertex labels (1-indexed).
    """

    def __init__(self, top_simplices, rng=None):
        super().__init__(top_simplices, dimension=3, rng=rng)

    def move_1_4(self, tet=None):
        """Perform a 1-4 Pachner move (stellar subdivision).

        Replace one tetrahedron with four by inserting a new vertex.
        Always valid.

        Parameters
        ----------
        tet : frozenset of int or None
            Tetrahedron to subdivide. If None, a random one is chosen.

        Returns
        -------
        bool
            Always True.
        """
        if tet is None:
            tet = self._rng.choice(list(self._simplices))

        self._simplices.discard(tet)
        v = self._new_vertex()

        for face in combinations(tet, 3):
            self._simplices.add(frozenset(face) | {v})

        return True

    def move_2_3(self, face=None):
        """Perform a 2-3 Pachner move.

        Replace two tetrahedra sharing a triangular face with three
        tetrahedra sharing an edge. Vertex count unchanged.

        Parameters
        ----------
        face : frozenset of int or None
            Shared triangular face. If None, a random valid face is
            chosen.

        Returns
        -------
        bool
            True if the move was performed, False if no valid face
            exists.
        """
        face_cofaces = self.face_to_cofaces(2)

        if face is None:
            valid_faces = []
            for f, cofaces in face_cofaces.items():
                if len(cofaces) != 2:
                    continue
                # the two opposite vertices
                d = next(iter(cofaces[0] - f))
                e = next(iter(cofaces[1] - f))
                # check that edge {d, e} is not in any tetrahedron
                if not self._edge_exists(d, e):
                    valid_faces.append(f)
            if not valid_faces:
                return False
            face = self._rng.choice(valid_faces)

        cofaces = face_cofaces[face]
        if len(cofaces) != 2:
            return False

        d = next(iter(cofaces[0] - face))
        e = next(iter(cofaces[1] - face))

        if self._edge_exists(d, e):
            return False

        # remove old tetrahedra
        self._simplices.discard(cofaces[0])
        self._simplices.discard(cofaces[1])

        # add new tetrahedra: for each edge of the face, create a
        # tet with the two opposite vertices
        for edge in combinations(face, 2):
            self._simplices.add(frozenset(edge) | {d, e})

        return True

    def move_3_2(self, edge=None):
        """Perform a 3-2 Pachner move (inverse of 2-3).

        Replace three tetrahedra sharing an edge with two tetrahedra
        sharing a face. Vertex count unchanged.

        Parameters
        ----------
        edge : frozenset of int or None
            The shared edge {d, e}. If None, a random valid edge is
            chosen.

        Returns
        -------
        bool
            True if the move was performed, False if no valid edge
            exists.
        """
        if edge is None:
            valid_edges = self._find_3_2_candidates()
            if not valid_edges:
                return False
            edge = self._rng.choice(valid_edges)

        d, e = tuple(edge)

        # find all tets containing this edge
        containing = [s for s in self._simplices if {d, e}.issubset(s)]
        if len(containing) != 3:
            return False

        # the link of the edge: vertices opposite to {d, e}
        link_verts = set()
        for s in containing:
            link_verts |= s - {d, e}

        if len(link_verts) != 3:
            return False

        # check the link forms a triangle (all 3 pairs appear as
        # faces of the containing tets)
        link_face = frozenset(link_verts)
        expected_tets = {
            frozenset({d, e}) | frozenset(pair)
            for pair in combinations(link_verts, 2)
        }
        # Defensive: with three distinct tets and a three-vertex link,
        # the tets are exactly the three {d, e} + pair combinations, so
        # this never fires. Kept as a guard against malformed input.
        if expected_tets != set(containing):  # pragma: no cover
            return False

        # check the new face {a, b, c} doesn't already exist as a
        # face of some other tetrahedron
        face_cofaces = self.face_to_cofaces(2)
        existing_cofaces = face_cofaces.get(link_face, [])
        # it should only appear in the tets we're about to remove
        if any(s not in expected_tets for s in existing_cofaces):
            return False

        # remove old tetrahedra
        for s in containing:
            self._simplices.discard(s)

        # add new tetrahedra
        self._simplices.add(link_face | {d})
        self._simplices.add(link_face | {e})

        return True

    def move_4_1(self, vertex=None):
        """Perform a 4-1 Pachner move (inverse of 1-4).

        Remove a vertex whose star consists of exactly 4 tetrahedra
        forming the boundary of a single tetrahedron.

        Parameters
        ----------
        vertex : int or None
            Vertex to remove. If None, a random valid vertex is
            chosen.

        Returns
        -------
        bool
            True if the move was performed, False if no valid vertex
            exists.
        """
        if vertex is None:
            valid_verts = self._find_4_1_candidates()
            if not valid_verts:
                return False
            vertex = self._rng.choice(valid_verts)

        # find all tets containing this vertex
        star = [s for s in self._simplices if vertex in s]
        if len(star) != 4:
            return False

        # the link: vertices opposite to v in each tet
        link_verts = set()
        for s in star:
            link_verts |= s - {vertex}

        if len(link_verts) != 4:
            return False

        # verify star is exactly the 4 tets of a subdivided tet
        new_tet = frozenset(link_verts)
        expected_star = {(new_tet - {w}) | {vertex} for w in link_verts}
        # Defensive: four distinct tets with a four-vertex link are
        # exactly the four faces of new_tet joined to ``vertex``, so
        # this never fires. Kept as a guard against malformed input.
        if expected_star != set(star):  # pragma: no cover
            return False

        # check the new tet doesn't already exist
        if new_tet in self._simplices:
            return False

        # remove old tetrahedra
        for s in star:
            self._simplices.discard(s)

        # add the collapsed tetrahedron
        self._simplices.add(new_tet)

        return True

    def _edge_exists(self, d, e):
        """Check if edge {d, e} exists in any tetrahedron."""
        return any({d, e}.issubset(s) for s in self._simplices)

    def _find_3_2_candidates(self):
        """Find all edges that are valid for a 3-2 move."""
        # count tets per edge
        edge_count = {}
        for s in self._simplices:
            for edge in combinations(s, 2):
                e = frozenset(edge)
                edge_count[e] = edge_count.get(e, 0) + 1

        face_cofaces = self.face_to_cofaces(2)

        candidates = []
        for edge, count in edge_count.items():
            if count != 3:
                continue
            # verify the link is a triangle
            d, e = tuple(edge)
            containing = [s for s in self._simplices if {d, e}.issubset(s)]
            link_verts = set()
            for s in containing:
                link_verts |= s - {d, e}
            if len(link_verts) != 3:
                continue

            link_face = frozenset(link_verts)
            expected_tets = {
                frozenset({d, e}) | frozenset(pair)
                for pair in combinations(link_verts, 2)
            }
            # Defensive: see move_3_2 — unreachable for well-formed input.
            if expected_tets != set(containing):  # pragma: no cover
                continue

            # check the new face isn't used elsewhere
            existing = face_cofaces.get(link_face, [])
            if any(s not in expected_tets for s in existing):
                continue

            candidates.append(edge)

        return candidates

    def _find_4_1_candidates(self):
        """Find all vertices valid for a 4-1 move."""
        # count tets per vertex
        vert_tets = {}
        for s in self._simplices:
            for v in s:
                vert_tets.setdefault(v, []).append(s)

        candidates = []
        for v, star in vert_tets.items():
            if len(star) != 4:
                continue
            link_verts = set()
            for s in star:
                link_verts |= s - {v}
            if len(link_verts) != 4:
                continue

            new_tet = frozenset(link_verts)
            expected = {(new_tet - {w}) | {v} for w in link_verts}
            # Defensive: see move_4_1 — unreachable for well-formed input.
            if expected != set(star):  # pragma: no cover
                continue
            if new_tet in self._simplices:
                continue

            candidates.append(v)

        return candidates

    def random_pachner_move(self, weights=None):
        """Apply a random Pachner move.

        Tries move types in weighted random order. Since move_1_4
        always succeeds, this is guaranteed to return True.

        Parameters
        ----------
        weights : tuple of float or None
            Weights for (1-4, 2-3, 3-2, 4-1). Default: equal.

        Returns
        -------
        bool
            True if a move was performed.
        """
        if weights is None:
            weights = (1.0, 1.0, 1.0, 1.0)

        moves = [
            self.move_1_4,
            self.move_2_3,
            self.move_3_2,
            self.move_4_1,
        ]
        # try each move type once in weighted random order
        indices = list(range(4))
        self._rng.shuffle(indices)
        indices.sort(
            key=lambda i: self._rng.random() * weights[i], reverse=True
        )
        for idx in indices:
            if moves[idx]():
                return True
        # move_1_4 always succeeds, so the loop always returns above.
        return False  # pragma: no cover