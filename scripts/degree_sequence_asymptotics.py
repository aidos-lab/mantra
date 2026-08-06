import json
import sys

import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from itertools import count

from joblib import delayed
from joblib import Parallel

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_samples


def one_skeleton(triangles):
    G = nx.Graph()
    for t in triangles:
        G.add_edges_from(combinations(t, 2))
    return G


def simplicial_complex(triangles):
    K = {0: set(), 1: set(), 2: set()}
    for t in triangles:
        for d in (1, 2, 3):
            for f in combinations(sorted(t), d):
                K[d - 1].add(f)
    return K


def euler_characteristic(triangles):
    K = simplicial_complex(triangles)
    return sum((-1) ** d * len(K[d]) for d in K)


def barycentric_subdivision(triangles):
    ctr = count(max((v for t in triangles for v in t), default=-1) + 1)
    bary = {}

    def b(simplex):
        key = tuple(sorted(simplex))
        if key not in bary:
            bary[key] = next(ctr)
        return bary[key]

    out = []
    for a, b_, c in triangles:
        f = b([a, b_, c])
        for u, v in [(a, b_), (b_, c), (a, c)]:
            m = b([u, v])
            out += [[f, m, u], [f, m, v]]

    return out


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

        mean_degree = []
        upper_bound = []

        for manifold in data:
            if manifold["name"] not in ["S^2", "T^2", "RP^2", "Klein bottle"]:
                continue

            K = manifold["triangulation"]
            K = barycentric_subdivision(K)

            G = one_skeleton(manifold["triangulation"])

            degree_sequence = sorted((d for _, d in G.degree()), reverse=True)

            mean_degree.append(np.mean(degree_sequence))
            upper_bound.append(
                6 - 6 * euler_characteristic(K) / len(degree_sequence)
            )

        sns.regplot(x=mean_degree, y=upper_bound)

        plt.show()
