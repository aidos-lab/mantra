import json
import sys

import numpy as np
import networkx as nx

from itertools import combinations
from scipy.stats import wasserstein_distance


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

        for manifold in data[:50]:
            G = one_skeleton(manifold["triangulation"])
            degree_sequence = sorted((d for _, d in G.degree()), reverse=True)
            degree_sequences.append(degree_sequence)

        D = pairwise_wasserstein(degree_sequences)
        print(D)
