from torch_geometric.transforms import BaseTransform

import numpy as np


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

    def __init__(self, normalize=False):
        """Create new moment curve embedding transform.

        Parameters
        ----------
        normalize : bool
            If set, normalize coordinates to a higher-dimensional
            sphere, thus increasing dimensionality by one.
        """
        super().__init__()

        self.normalize = normalize

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

        n = data["n_vertices"].item()
        d = data["dimension"].item()

        X = _calculate_moment_curve(n, d)

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
            Z = np.sqrt(1 - Z**2)
            X = np.column_stack((X, Z))

        data["moment_curve_embedding"] = X

        return data
