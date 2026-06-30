import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from collections import defaultdict

#TODO: Looka at what this does
def print_statistics(dataset):
    """Print per-class statistics of a dataset.

    Parameters
    ----------
    dataset : list of dict
        Dataset entries.
    """
    class_entries = defaultdict(list)
    for entry in dataset:
        class_entries[entry["name"]].append(entry)

    print(
        f"{'Class':<30} {'Count':>8} {'Min V':>8} {'Mean V':>8} {'Max V':>8}"
    )
    print("-" * 56)
    for name in sorted(class_entries.keys()):
        entries = class_entries[name]
        nverts = [e["n_vertices"] for e in entries]
        print(
            f"{name:<30} {len(entries):>8} "
            f"{min(nverts):>8} {round(np.mean(nverts),2):>8} {max(nverts):>8}"
        )
    print(f"\nTotal: {len(dataset)}")

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data).set_index("id")
    df = df.drop(
        columns=[
            "triangulation",
            "torsion_coefficients",
            "betti_numbers",
            "orientable",
            "dimension",
        ]
    )

    # So sue me, right?
    interesting_manifolds = ["RP^2", "T^2", "Klein bottle", "S^2"]
    print(f"All manifold types: {df['name'].unique()}")

    df = df.query("name in @interesting_manifolds")
    df = df.groupby(["name", "n_vertices"]).size().to_frame("count")

    sns.set_theme(style="white", palette="Set1")

    g = sns.lineplot(df, x="n_vertices", y="count", hue="name")
    g.set(yscale="log")

    plt.tight_layout()
    plt.show()
