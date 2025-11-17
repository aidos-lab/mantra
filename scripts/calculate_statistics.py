import json
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

    df = df.query("name in @interesting_manifolds")
    df = df.groupby(["name", "n_vertices"]).size().to_frame("count")

    sns.set_theme(style="white", palette="Set1")

    g = sns.lineplot(df, x="n_vertices", y="count", hue="name")
    g.set(yscale="log")

    plt.tight_layout()
    plt.show()
