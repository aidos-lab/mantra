import json
import sys

import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from itertools import count

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import wasserstein_distance

from sklearn.metrics import silhouette_samples


def pairwise_wasserstein(sequences):
    n = len(sequences)
    D = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        D[i, j] = D[j, i] = wasserstein_distance(sequences[i], sequences[j])
    return D


def one_skeleton(triangles):
    G = nx.Graph()
    for t in triangles:
        G.add_edges_from(combinations(t, 2))
    return G


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

        for manifold in data:
            if manifold["name"] not in ["S^2", "T^2", "RP^2", "Klein bottle"]:
                continue

            if len(degree_sequences) == 200:
                break

            K = manifold["triangulation"]
            K.extend(barycentric_subdivision(K))

            G = one_skeleton(manifold["triangulation"])

            degree_sequence = sorted((d for _, d in G.degree()), reverse=True)
            degree_sequences.append(degree_sequence)

            labels.append(manifold["name"])

        D = pairwise_wasserstein(degree_sequences)

        sil = silhouette_samples(D, labels, metric="precomputed")
        df = pd.DataFrame({"label": labels, "silhouette": sil})

        print(df.groupby("label")["silhouette"].mean().sort_values())

        C = squareform(D, checks=False)
        Z = linkage(C, method="average")

        sns.clustermap(
            D,
            row_linkage=Z,
            col_linkage=Z,
            xticklabels=labels,
            yticklabels=labels,
            annot=False,
            fmt=".2f",
        )

        plt.show()
