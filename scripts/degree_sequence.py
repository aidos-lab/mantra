import json
import sys

import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

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


def loo_nn_predict(D, labels):
    labels = np.asarray(labels)
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)

    # If there are duplicate values, we are still assigning the label of
    # the neighbor with the closest index. We do not want that, so let's
    # rather take a majority vote.
    row_min = Dm.min(axis=1, keepdims=True)
    ties = np.isclose(Dm, row_min)

    preds = np.empty(len(labels), dtype=labels.dtype)
    for i in range(len(labels)):
        preds[i] = Counter(labels[ties[i]]).most_common(1)[0][0]

    return preds


def penalty_matching_distance(x, y, penalty, p=1):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, m = len(x), len(y)
    C = np.zeros((n + m, n + m))

    C[:n, :m] = np.abs(x[:, None] - y[None, :]) ** p
    C[:n, m:] = penalty**p
    C[n:, :m] = penalty**p

    r, c = linear_sum_assignment(C)
    return C[r, c].sum() ** (1 / p)


def pairwise_parallel(sequences, metric, n_jobs=-1):
    n = len(sequences)
    pairs = list(combinations(range(n), 2))
    vals = Parallel(n_jobs=n_jobs)(
        delayed(metric)(sequences[i], sequences[j]) for i, j in pairs
    )

    D = np.zeros((n, n))
    for (i, j), v in zip(pairs, vals):
        D[i, j] = D[j, i] = v

    return D


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

        degree_sequences = []
        labels = []
        chi = []

        for manifold in data:
            if manifold["name"] not in ["S^2", "T^2", "RP^2", "Klein bottle"]:
                continue

            if len(degree_sequences) == 1500:
                break

            K = manifold["triangulation"]
            K = barycentric_subdivision(K)

            G = one_skeleton(manifold["triangulation"])

            degree_sequence = sorted((d for _, d in G.degree()), reverse=True)
            degree_sequences.append(degree_sequence)

            labels.append(manifold["name"])
            chi.append(euler_characteristic(K))

        D1 = pairwise_parallel(
            degree_sequences, lambda x, y: wasserstein_distance(x, y)
        )

        D2 = pairwise_parallel(
            degree_sequences,
            lambda x, y: penalty_matching_distance(x, y, 6, p=1),
        )

        D3 = pairwise_parallel(
            chi,
            lambda x, y: np.abs(x - y),
        )

        for D, name in zip(
            [D1, D2, D3],
            ["Wasserstein", "Partial matching", "Euler characteristic"],
        ):
            print("Metric:", name)
            sil = silhouette_samples(D, labels, metric="precomputed")
            df = pd.DataFrame({"label": labels, "silhouette": sil})
            print(df.groupby("label")["silhouette"].mean().sort_values())

            pred = loo_nn_predict(D, labels)
            print(confusion_matrix(labels, pred))
            print(classification_report(labels, pred, zero_division=0.0))

        C = squareform(D1, checks=False)
        Z = linkage(C, method="average")

        sns.clustermap(
            D1,
            row_linkage=Z,
            col_linkage=Z,
            xticklabels=labels,
            yticklabels=labels,
            annot=False,
            fmt=".2f",
        )

        plt.show()
