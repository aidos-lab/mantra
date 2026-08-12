import json
import random
import sys

import numpy as np
import networkx as nx
import pandas as pd

from collections import Counter

from itertools import combinations
from itertools import count
from itertools import permutations

from joblib import delayed
from joblib import Parallel

from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_samples

pd.options.styler.format.precision = 2
pd.set_option("display.precision", 2)


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


def one_skeleton(simplices):
    G = nx.Graph()
    for s in simplices:
        G.add_edges_from(combinations(s, 2))
    return G


def simplicial_complex(simplices):
    K = {}
    for s in simplices:
        for d in range(1, len(s) + 1):
            K.setdefault(d - 1, set())
            for f in combinations(sorted(s), d):
                K[d - 1].add(f)
    return K


def euler_characteristic(simplices):
    K = simplicial_complex(simplices)
    return sum((-1) ** d * len(K[d]) for d in K)


def barycentric_subdivision(simplices):
    ctr = count(max((v for s in simplices for v in s), default=-1) + 1)
    bary = {}

    def b(simplex):
        key = tuple(sorted(simplex))
        if key not in bary:
            bary[key] = next(ctr)
        return bary[key]

    out = []
    for s in simplices:
        for perm in permutations(s):
            out.append([b(perm[: i + 1]) for i in range(len(perm))])

    return out


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

        degree_sequences = []
        labels = []
        chi = []

        data = list(
            filter(
                lambda manifold: manifold["name"]
                in ["S^2", "T^2", "RP^2", "Klein bottle"],
                data,
            )
        )

        data = random.sample(data, 1000)

        for manifold in data:
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

            report = classification_report(
                labels, pred, zero_division=0.0, output_dict=True
            )

            df = pd.DataFrame.from_dict(report).transpose()
            df["support"] = df["support"].astype(int)

            print(df)

            print(
                df.style.to_latex(
                    hrules=True,
                    column_format="lSSSS",
                )
            )
