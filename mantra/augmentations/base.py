"""Base class for mutable triangulation data structures."""

import random
from collections import defaultdict
from itertools import combinations, permutations


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

    def stellar_subdivide(self, simplex=None):
        """Stellar-subdivide one top-dimensional simplex.

        Insert a new barycenter vertex and replace ``simplex`` with its
        ``d + 1`` codimension-1 faces, each joined to the new vertex. This
        is the 1-(d+1) Pachner move (the 1-3 move in 2D, the 1-4 move in
        3D) and is always valid.

        Parameters
        ----------
        simplex : frozenset of int or None
            Simplex to subdivide. If None, a random top-simplex is chosen.

        Returns
        -------
        bool
            Always True.
        """
        if simplex is None:
            simplex = self._rng.choice(list(self._simplices))

        self._simplices.discard(simplex)
        v = self._new_vertex()

        for face in combinations(simplex, len(simplex) - 1):
            self._simplices.add(frozenset(face) | {v})

        return True

    def stellar_subdivision(self, fraction=1.0):
        """Stellar-subdivide a (random) fraction of top-dim simplices.

        Each chosen simplex is replaced via :meth:`stellar_subdivide`;
        skipped simplices pass through unchanged. Their boundaries are
        untouched, so partial application preserves the simplicial-complex
        condition across shared codimension-1 faces.

        Parameters
        ----------
        fraction : float
            Fraction of top-dim simplices to subdivide, in [0, 1]. 1.0
            (default) subdivides all of them.

        Raises
        ------
        ValueError
            If ``fraction`` is outside [0, 1].
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")

        simplices = list(self._simplices)
        if fraction == 1.0:
            chosen = simplices
        else:
            n_subdivide = round(fraction * len(simplices))
            # A positive fraction must subdivide at least one simplex;
            # otherwise a small fraction silently rounds to a no-op.
            if fraction > 0.0 and simplices:
                n_subdivide = max(1, n_subdivide)
            chosen = self._rng.sample(simplices, n_subdivide)

        for simplex in chosen:
            self.stellar_subdivide(simplex)

    def barycentric_subdivision(self):
        """Apply full barycentric subdivision (the order complex).

        Each sub-simplex of each top-dimensional simplex becomes a new
        vertex, and each permutation of a top simplex's vertices yields a
        new top simplex (its prefix chain). Replaces the triangulation in
        place; vertex labels are reassigned.
        """
        all_simplices = {}
        next_vertex = 1
        for simplex in self._simplices:
            simplex_sorted = sorted(simplex)
            for k in range(1, len(simplex_sorted) + 1):
                for face in combinations(simplex_sorted, k):
                    key = frozenset(face)
                    if key not in all_simplices:
                        all_simplices[key] = next_vertex
                        next_vertex += 1

        new_simplices = set()
        for simplex in self._simplices:
            simplex_sorted = sorted(simplex)
            for perm in permutations(simplex_sorted):
                chain = (
                    all_simplices[frozenset(perm[:k])]
                    for k in range(1, len(perm) + 1)
                )
                new_simplices.add(frozenset(chain))

        self._simplices = new_simplices
        self._next_vertex = next_vertex

    def graded_subdivision(self, target_vertices):
        """Stellar-subdivide simplices until reaching a vertex target.

        Repeatedly picks a top-simplex at random and stellar-subdivides it,
        cycling through the complex (each simplex of the current generation
        is subdivided once before any of its children is), until the
        triangulation has ``target_vertices`` vertices. Each stellar move
        adds exactly one vertex, so the target is reached exactly when it is
        above the current vertex count.

        Parameters
        ----------
        target_vertices : int
            Number of vertices to grow the triangulation to. Must be
            strictly greater than the current vertex count, since the
            purpose is to *refine* the triangulation into unseen data.

        Raises
        ------
        ValueError
            If ``target_vertices`` is not strictly above the current
            vertex count: stellar subdivision only adds vertices, so a
            target at or below the current count cannot be reached.
        """
        n = self.n_vertices
        if target_vertices <= n:
            raise ValueError(
                f"target_vertices ({target_vertices}) must be strictly "
                f"greater than the current vertex count ({n}); graded "
                f"subdivision only adds vertices."
            )
        # Simplices of the current generation still awaiting subdivision;
        # children produced this pass are not added until the next pass.
        pending = set(self._simplices)

        while n < target_vertices:
            if not pending:
                pending = set(self._simplices)
            chosen = self._rng.choice(list(pending))
            pending.discard(chosen)
            self.stellar_subdivide(chosen)
            n += 1

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
