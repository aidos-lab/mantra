from torch_geometric.transforms import BaseTransform

import numpy as np


def _calculate_moment_curve(n, d):
    """Calculate moment curve for `n` vertices of a `d`-dimensional manifold.

    This is an auxiliary function for calculating the moment curve of
    `n` vertices of a `d`-dimensional manifold. Notice that the curve
    will have coordinates of dimension `2d + 1`.

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
    """
    t = np.arange(n, dtype=float)
    t /= (n - 1)

    X = np.vstack([t**k for k in range(1, 2 * d + 2)]).T
    return X


class MomentCurveEmbedding(BaseTransform):

    def forward(self, data):
        print(data.keys())
        n = data["n_vertices"].item()
        d = data["dimension"].item()

        X = _calculate_moment_curve(n, d)
        print(X)

        return data

    def spherical_moment_curve(n, d):
        t = np.linspace(0, 1, n, endpoint=False)
        t = t + 0.5 / n

        p = np.vstack([t**k for k in range(1, 2 * d + 1)]).T

        return p

        #norms = np.linalg.norm(p, axis=1)
        #q = p / norms.max()  # scale so all norms ≤ 1

        #rq = np.linalg.norm(q, axis=1)

        #z = np.sqrt(1.0 - rq**2)  # last coord to land on S^5
        #return np.hstack([q, z[:, None]])  # shape (n, 6)

