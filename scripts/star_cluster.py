from itertools import combinations

import json
import sys


def build_complex(top_simplices):
    K = set()
    for f in top_simplices:
        f = tuple(f)
        for k in range(1, len(f) + 1):
            K.update(frozenset(c) for c in combinations(f, k))
    return K


def closed_star(K, v):
    return {s for s in K if s | {v} in K}


def open_star(K, v):
    return {s for s in K if v in s}


def star_cluster(K, sigma):
    return set().union(*(closed_star(K, v) for v in sigma)) if sigma else set()


def link(K, sigma):
    sigma = frozenset(sigma)
    return {tau for tau in K if tau.isdisjoint(sigma) and (tau | sigma) in K}


def facets(K):
    out = []
    for s in sorted(K, key=len, reverse=True):
        if not any(s < m for m in out):
            out.append(s)
    return out


def vertices(K):
    out = []
    for s in K:
        if len(s) == 1:
            out.append(list(s)[0])
    return out


def euler_characteristic(K):
    return sum((-1) ** (len(s) - 1) for s in K)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

        for manifold in data[:10]:
            K = build_complex(manifold["triangulation"])

            print(manifold["name"])

            maybe_collapsible = 0
            not_collapsible = 0

            for facet in facets(K):
                L = star_cluster(K, facet)
                V = [s for s in facet]

                allowed_vertices = vertices(L)
                subcomplex = set()
                for sigma in K:
                    if all(v in allowed_vertices for v in sigma):
                        subcomplex.add(sigma)

                for v in vertices(L):
                    if v not in V:
                        ell = link(subcomplex, {v})
                        chi = euler_characteristic(ell)

                        if chi != 1:
                            not_collapsible += 1
                        else:
                            maybe_collapsible += 1

            print("\t", maybe_collapsible, not_collapsible)
