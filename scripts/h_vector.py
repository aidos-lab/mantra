from itertools import chain
from itertools import combinations
from itertools import permutations

from math import comb


def _powerset(s):
    s = list(s)
    return chain.from_iterable(combinations(s, k) for k in range(len(s) + 1))


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


def h_vector(top_simplices):
    d = len(top_simplices[0])
    fv = f_vector(top_simplices)
    f = [1] + list(fv) + [0] * (d - len(fv))
    return [
        sum((-1) ** (k - i) * comb(d - i, k - i) * f[i] for i in range(k + 1))
        for k in range(d + 1)
    ]


def local_h_vector(sigma):
    sigma = tuple(sorted(sigma))
    d = len(sigma)
    ell = [0] * (d + 1)

    for tau in _powerset(sigma):
        h = [1] if len(tau) == 0 else h_vector(barycentric_subdivision([tau]))
        sign = (-1) ** (d - len(tau))
        for k, hk in enumerate(h):
            ell[k] += sign * hk

    return ell


def all_local_h_vectors(top_simplices):
    return {f: local_h_vector(f) for f in faces(top_simplices)}
