import json
import sys

import numpy as np

from collections import Counter

from itertools import combinations
from itertools import permutations


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix


def rank_selected_subcomplex(top_simplices, coloring, S):
    S = set(S)
    return [
        f for f in faces(top_simplices) if all(coloring[v] in S for v in f)
    ]


def reduced_euler_characteristic(simplices):
    if not simplices:
        return -1
    return sum((-1) ** (len(simplices) - 1) for s in simplices) - 1


def rank_selected_euler_characteristics(top_simplices, coloring):
    d = max(coloring.values())
    colors = range(d + 1)

    result = {}
    for size in range(d + 2):
        for S in combinations(colors, size):
            simplices = rank_selected_subcomplex(top_simplices, coloring, S)
            result[S] = reduced_euler_characteristic(simplices)

    return result


def faces(top_simplices):
    F = set()

    for s in top_simplices:
        s = tuple(sorted(s))
        for k in range(1, len(s) + 1):
            F.update(combinations(s, k))

    return F


def vertices(top_simplices):
    result = set()
    for s in top_simplices:
        result |= set(s)
    return result


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


def colored_barycentric_subdivision(top_simplices):
    new_top_simplices = barycentric_subdivision(top_simplices)

    new_vertices = vertices(new_top_simplices)
    coloring = {v: len(v) - 1 for v in new_vertices}

    return new_top_simplices, coloring


def invariant_to_vector(invariant, d):
    colors = range(d + 1)
    subsets = [S for size in range(d + 2) for S in combinations(colors, size)]
    return [invariant[S] for S in subsets], subsets


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

    confused_pairs = Counter()

    X = []

    for manifold in data:
        K = manifold["triangulation"]
        top_simplices, coloring = colored_barycentric_subdivision(K)

        invariant = rank_selected_euler_characteristics(
            top_simplices, coloring
        )

        x, _ = invariant_to_vector(invariant, d=manifold["dimension"])
        X.append(x)

    X = np.asarray(X)
    y = [manifold["name"] for manifold in data]

    clf = RandomForestClassifier(random_state=42)
    y_pred = cross_val_predict(
        clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42)
    )

    cm = confusion_matrix(y, y_pred, labels=sorted(set(y)))
    print(sorted(set(y)))
    print(cm)

    print(
        f"{100 * accuracy_score(y, y_pred):.02f}%",
        " / " f"{100 * balanced_accuracy_score(y, y_pred):.02f}%",
    )
