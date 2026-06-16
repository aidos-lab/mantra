"""2D triangulation with Pachner moves and topology-changing operations."""

from itertools import combinations

from mantra.augmentations.base import Triangulation

# Minimal triangulation of the torus (7 vertices, 14 triangles).
# Removing triangle {1, 2, 3} leaves 13 triangles with boundary {1,2,3}.
_TORUS_TRIANGULATION_MINUS_FACE = [
    [1, 2, 4],
    [1, 3, 5],
    [1, 4, 6],
    [1, 5, 7],
    [1, 6, 7],
    [2, 3, 6],
    [2, 4, 7],
    [2, 5, 6],
    [2, 5, 7],
    [3, 4, 5],
    [3, 4, 7],
    [3, 6, 7],
    [4, 5, 6],
]

# Minimal triangulation of RP^2 (6 vertices, 10 triangles).
# This is the hemicosahedron. Removing triangle {1, 2, 3} leaves
# 9 triangles with boundary vertices {1, 2, 3} and interior
# vertices {4, 5, 6}.
_RP2_TRIANGULATION_MINUS_FACE = [
    [1, 2, 4],
    [1, 3, 6],
    [1, 4, 5],
    [1, 5, 6],
    [2, 3, 5],
    [2, 4, 6],
    [2, 5, 6],
    [3, 4, 5],
    [3, 4, 6],
]


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
