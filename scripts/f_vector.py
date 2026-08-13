import math
import json
import random
import sys

import numpy as np

from itertools import combinations
from itertools import permutations

from scipy.special import stirling2


def faces(top_simplices):
    F = set()

    for s in top_simplices:
        s = tuple(sorted(s))
        for k in range(1, len(s) + 1):
            F.update(combinations(s, k))

    return F


def f_vector(top_simplices):
    count = {}

    for f in faces(top_simplices):
        count[len(f) - 1] = count.get(len(f) - 1, 0) + 1

    return [count.get(i, 0) for i in range(max(count) + 1)]


def barycentric_subdivision(top_simplices):
    new = set()

    for s in top_simplices:
        s = tuple(sorted(s))
        for p in permutations(s):
            new.add(
                tuple(
                    sorted(tuple(sorted(p[:k])) for k in range(1, len(p) + 1))
                )
            )

    return sorted(new)


def euler_characteristic(top_simplices):
    return sum((-1) ** (len(f) - 1) for f in faces(top_simplices))


def brenti_welker_matrix(dim):
    d = dim + 1

    return np.asarray(
        [
            [math.factorial(i + 1) * stirling2(j + 1, i + 1) for j in range(d)]
            for i in range(d)
        ],
        float,
    )


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

        # FIXME: Should make this configurable since it only applies to
        # dimension 2. Maybe get a class count and only pick *some*?
        data = list(
            filter(
                lambda manifold: manifold["name"]
                in ["Klein bottle", "RP^2", "S^2", "T^2"],
                data,
            )
        )

    dim = [manifold["dimension"] for manifold in data]
    assert min(dim) == max(dim), "Require same dimension"
    dim = dim[0]

    M = brenti_welker_matrix(dim)
    eigenvalues, eigenvectors = np.linalg.eig(M.T)

    data = random.sample(data, 500)

    for manifold in data:
        K = manifold["triangulation"]
        L = barycentric_subdivision(K)

        x = f_vector(K)
        y = f_vector(L)

        print(
            manifold["name"],
            x,
            euler_characteristic(K),
            y,
            euler_characteristic(L),
        )

        # Let's check that we are doing the right thing: The matrix
        # helps us get the right f-vector of a *single* subdivision
        # step.
        assert np.allclose(np.dot(M, x) - y, 0)

        c_K = eigenvectors @ x
        c_L = eigenvectors @ y

        print(c_K, c_L)
