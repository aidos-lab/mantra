from torch_geometric.transforms import BaseTransform

from itertools import combinations

import numpy as np
import torch


def _calculate_moment_curve(n, d):
    """Calculate moment curve for `n` vertices of a `d`-dimensional manifold.

    This is an auxiliary function for calculating the moment curve of
    `n` vertices of a `d`-dimensional manifold. Notice that the curve
    will have coordinates of dimension `2d + 1`.

    The moment curve is a canonical representation of a triangulation
    but its coordinates are by necessity high-dimensional, and merely
    making use of the number of vertices and the dimension. This will
    mean that the coordinates, by themselves, are not enough to fully
    characterize a triangulation (which is a good thing).

    Parameters
    ----------
    n : int
        Number of vertices

    d : int
        Dimension of the manifold

    Returns
    -------
    np.array of shape (n, 2 * d + 1)
        Coordinates of vertices on the moment curve. Coordinates are
        float values.

    Notes
    -----
    This function is implemented inspired by an article of Francesco
    Mezzadri [1]_.

    References
    ----------
    .. [1] Francesco Mezzadri, "How to Generate Random Matrices from the
    Classical Compact Groups," Notices of the American Mathematical
    Society, Vol. 54, pp. 592--604, 2007.
    """
    t = np.arange(n, dtype=float)
    t /= n - 1

    X = np.vstack([t**k for k in range(1, 2 * d + 2)]).T
    return X


def _propagate_values(X, triangulation):
    """Propagate vertex-based values to all simplices.

    This helper function propagates vertex-based values to all simplices
    by calculating the respective barycenter. That is, given any simplex
    of dimension > 0, we will calculate the barycenter of its respective
    values at the vertices.

    Parameters
    ----------
    X : np.array of shape (n, d)
        Vertex-based attributes

    triangulation : list of lists
        A triangulation, expressed as a list of top-level simplices,
        which themselves are lists (or iterable; we do not actually
        care here).

    Returns
    -------
    dict of np.array of shape (n_k, d)
        A dictionary whose keys indicate the respective zero-indexed
        dimension and whose values are the respective values for all
        simplices of that dimension (ordered lexicographically).
    """
    simplices = set([tuple(s) for s in triangulation])
    max_dim = len(next(iter(simplices)))

    for simplex in triangulation:
        for dim in range(1, max_dim):
            simplices.update(s for s in combinations(simplex, r=dim))

    # To sort lexicographically, we need to turn this back into
    # something mutable.
    simplices = list(simplices)
    simplices.sort()
    simplices.sort(key=len)

    values = {
        0: X,
    }

    for dim in range(1, max_dim):
        simplices_ = [s for s in simplices if len(s) == dim + 1]
        M = []

        for s in simplices_:
            # View as an array to correct for the index shift; our
            # triangulation is not zero-indexed.
            s = np.asarray(s)

            # Calculate barycenter for the current simplex (i.e., one
            # row of the result matrix).
            M.append(np.mean(X[s - 1, :], axis=0))

        M = np.asarray(M)
        values[dim] = torch.from_numpy(M).to(torch.float32)

    return values


def _sample_from_special_orthogonal_group(n, rng=None):
    """Generate a sample (a matrix) from SO(n), the special orthogonal group.

    This function calculates an element of SO(n), the special orthogonal
    group in dimension `n`.

    Parameters
    ----------
    n : int
        Dimension of the desired matrix. Negative numbers are not
        accepted.

    rng : np.random.Generator or None
        Random number generator object. If set to `None`, the default
        generator (`np.random.default_rng()`) will be used.

    Returns
    -------
    np.array of shape (n, n)
        A random element of SO(n)
    """
    assert n > 0, "Positive dimension is required"

    if rng is None:
        rng = np.random.default_rng()

    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)

    # Flip the sign of the diagonal entries to be positive. This is
    # numerically more stable than the Mezzadri's approach.
    s = np.sign(np.diag(R))
    s[s == 0] = 1
    Q = Q * s

    # Enforce a positive determinant by optionally flipping the sign of
    # the first column.
    Q[:, 0] *= np.sign(np.linalg.det(Q))

    return Q


class MomentCurveEmbedding(BaseTransform):

    def __init__(
        self, perturb=False, normalize=False, propagate=False, rng=None
    ):
        """Create new moment curve embedding transform.

        Parameters
        ----------
        perturb : bool
            If set, perturbs all coordinates by applying a random
            rotation to them. This has the effect of rotating the
            triangulation but does not change any distances, thus
            making any tasks operating with triangulations harder
            and ensuring that all manifolds will have nigh-unique
            coordinates.

        normalize : bool
            If set, normalize coordinates to a higher-dimensional
            sphere, thus increasing dimensionality by one.
        propagate : bool
            If set propagates the values upwards from the 0-simplices
            to the k-simplices above by getting a barycenter. Note
            this option requires that `triangulation` be present
            in the data object.

        rng : np.random.Generator, int, or None
            Random number generator object. If set to `None`, the
            default generator (`np.random.default_rng()`) will be
            used. If set to an int, it will be used as a seed for
            a new `np.random.default_rng()` instance.
        """
        super().__init__()

        self.propagate = propagate
        self.perturb = perturb
        self.normalize = normalize

        self.rng = rng if rng is not None else np.random.default_rng()

        # Let's see whether we are actually a random number generator or
        # not...
        if isinstance(self.rng, int):
            self.rng = np.random.default_rng(seed=self.rng)

    def forward(self, data):
        """Calculate moment curve embedding for a data object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object

        Returns
        -------
        torch_geometric.data.Data
            Data object with a new `moment_curve_embedding` key added.
            The attribute will be overwritten if already present.
        """
        assert "n_vertices" in data and "dimension" in data

        n = data["n_vertices"]
        d = data["dimension"]

        X = _calculate_moment_curve(n, d)

        if self.perturb:
            Q = _sample_from_special_orthogonal_group(X.shape[1], rng=self.rng)
            X = X @ Q.T

        if self.normalize:
            # 1. We first get the "original" norms of points on the
            #    moment curve.
            #
            # 2. We re-normalize so that the largest norm is *one*.
            #    Notice that this is *not* a proper projection yet.
            #
            # 3. We calculate the difference to a unit-norm vector,
            #    thus figuring out what the *new* coordinate should
            #    be so that the point is on the sphere.
            #
            #  4. We concatenate the new coordinate, thus *finally*
            #     putting everything on the sphere.
            norms = np.linalg.norm(X, axis=1)
            X = X / norms.max()
            Z = np.linalg.norm(X, axis=1)
            Z = np.sqrt(np.maximum(1 - Z**2, 0.0))
            X = np.column_stack((X, Z))

        # NOTE:: This fixes the case where we already performed a mapping from a
        # simplicial complex to a graph and want to get an embedding for
        # whatever the nodes are, however this leaves open
        # the case where we might want to embedding the simplicial complex
        # and then map the embedding through the conversion
        if self.propagate:
            assert (
                "triangulation" in data
            ), "Data object must contain `triangulation` to perform propagation"
            data["moment_curve_embedding"] = _propagate_values(
                X, data["triangulation"]
            )
        else:
            data["moment_curve_embedding"] = torch.from_numpy(X).to(
                torch.float32
            )

        return data
