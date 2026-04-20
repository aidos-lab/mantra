"""Base class for mutable triangulation data structures."""

import random
from collections import defaultdict
from itertools import combinations


class Triangulation:
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
        return sorted(
            sorted(remap[v] for v in s) for s in self._simplices
        )

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
