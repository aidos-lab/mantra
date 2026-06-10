"""3D triangulation with Pachner moves."""

from itertools import combinations

from mantra.augmentations.base import Triangulation


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
        if expected_tets != set(containing):
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
        if expected_star != set(star):
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
            if expected_tets != set(containing):
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
            if expected != set(star):
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
        return False
