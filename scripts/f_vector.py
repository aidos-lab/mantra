import math
import json
import random
import sys

import numpy as np

from collections import Counter

from itertools import combinations
from itertools import permutations
from itertools import pairwise

from scipy.special import stirling2
from scipy.linalg import eig


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
        int,
    )


def project(f_vector, eigenvalues, eigenvectors):
    # Project the f-vector into eigencoordinates. This assumes that the
    # eigenvectors are the *left* eigenvectors and that the i-th vector
    # is in the i-th row.
    c = eigenvectors @ f_vector

    # Since eigenvalues are sorted we know that the *last* one is the
    # largest. We may now normalize all other values by this. The log
    # below is safe since all eigenvalues are nonzero integers.
    top = eigenvalues.size - 1
    alpha = np.log(eigenvalues) / np.log(eigenvalues[top])
    c_normalized = c / (c[top] ** alpha)

    # In case the Euler characteristic carries some signal, we can use
    # it to distinguish things a bit better.
    chi = sum((-1) ** i * f_vector[i] for i in range(len(f_vector)))
    return np.concatenate([[chi], c_normalized])


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

        # FIXME: Should make this configurable since it only applies to
        # dimension 2. Maybe get a class count and only pick *some*?
        if data[0]["dimension"] == 2:
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

    # Set up the transformation matrix and get its eigenvalues and left
    # eigenvectors. I take the eigenvalues from the diagonal of the `M`
    # because it is a triangular matrix and there is no need to cast it
    # to `float`. We need the *left* eigenvectors because we want to be
    # able to *project* into a basis of eigenvectors.
    M = brenti_welker_matrix(dim)
    eigenvalues, eigenvectors = eig(M, left=True, right=False)

    # Minor subtlety: The eigenvalues could be ordered differently; the
    # safe thing is to check for this explicitly. As a simple trick, we
    # can just order the real parts of the eigenvalues, then get a mask
    # to reorder the _precise_ eigenvalues (and eigenvectors).
    indices = np.argsort(eigenvalues.real)

    eigenvalues = np.diag(M)[indices]
    eigenvectors = eigenvectors[:, indices]

    # Minor subtlety: We get *columns* back from `eig` above, but for a
    # projection to work, the eigenvectors need to be in the rows.
    eigenvectors = eigenvectors.T
    eigenvectors = eigenvectors / eigenvectors.max(axis=1, keepdims=1)

    confused_pairs = Counter()

    for manifold1, manifold2 in pairwise(data):
        K = manifold1["triangulation"]
        L = manifold2["triangulation"]

        x = f_vector(K)
        y = f_vector(L)

        out = ""
        out += f"{manifold1['name']} {x}"
        out += " vs. "
        out += f"{manifold2['name']} {y}"

        a = project(x, eigenvalues, eigenvectors)
        b = project(y, eigenvalues, eigenvectors)

        delta = np.linalg.norm(a - b)
        expect_zero = False

        if manifold1["name"] == manifold2["name"]:
            expect_zero = True

        if (expect_zero and np.isclose(delta, 0.0)) or (
            not expect_zero and not np.isclose(delta, 0.0)
        ):
            out = "✅ " + out
        else:
            out = "❌ " + out + f" (delta = {delta:.02f})"

            n1 = manifold1["name"]
            n2 = manifold2["name"]

            if n1 > n2:
                n1, n2 = n2, n1

            confused_pairs[(n1, n2)] += 1

        print(out)

    print("")

    for n1, n2 in confused_pairs:
        print(
            f"{n1} vs. {n2}:",
            confused_pairs[(n1, n2)],
            "failures",
            f"({len(data) - 1} items)",
        )
