import json
import sys

import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations

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
        G.add_edges_from(combinations(t, 2))  # the 3 edges of the triangle
    return G


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

        degree_sequences = []
        labels = []

        for manifold in data:
            if manifold["name"] not in ["S^2", "T^2", "RP^2", "Klein bottle"]:
                continue

            if len(degree_sequences) == 100:
                break

            G = one_skeleton(manifold["triangulation"])

            degree_sequence = sorted((d for _, d in G.degree()), reverse=True)
            degree_sequences.append(degree_sequence)

            labels.append(manifold["name"])

        D = pairwise_wasserstein(degree_sequences)

        sil = silhouette_samples(D, labels, metric="precomputed")
        df = pd.DataFrame({"label": labels, "silhouette": sil})

        print(df.groupby("label")["silhouette"].mean().sort_values())

        print(sil)

        C = squareform(D, checks=False)
        Z = linkage(C, method="average")

        sns.clustermap(
            D,
            row_linkage=Z,
            col_linkage=Z,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".2f",
        )

        plt.show()
