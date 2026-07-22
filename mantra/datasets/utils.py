from collections import Counter

import numpy as np
import requests
from sklearn.model_selection import train_test_split


def _get_mantra_dataset_url(
    version: str, dimension: int, balanced: bool = False
) -> str:
    """Get URL to download dataset from."""
    suffix = "_balanced" if balanced else ""
    filename = f"{dimension}_manifolds{suffix}.json.gz"

    if version == "latest":
        return f"https://github.com/aidos-lab/MANTRA/releases/latest/download/{filename}"  # noqa

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(
        "https://api.github.com/repos/aidos-lab/mantra/releases",
        headers=headers,
    )

    all_available_versions = [item["name"] for item in response.json()]

    if version not in all_available_versions:
        raise ValueError(
            f"Version {version} not available, please choose one of the following versions: {all_available_versions}."  # noqa
        )

    # Note that the URL order is different and thus inconsistent for a
    # specific release.
    return f"https://github.com/aidos-lab/MANTRA/releases/download/{version}/{filename}"  # noqa


def filter_by_class_count(entries, label_source, min_count):
    """Drop entries whose ``label_source`` value occurs <= ``min_count`` times.

    A ``min_count`` of ``None`` (or <= 0) disables filtering.
    """
    if min_count is None or min_count <= 0:
        return entries, Counter()
    counts = Counter(e[label_source] for e in entries)
    kept_labels = {lbl for lbl, c in counts.items() if c > min_count}
    filtered = [e for e in entries if e[label_source] in kept_labels]
    return filtered, counts


def make_split_index(
    data_list_size: int,
    seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
    labels=None,
):
    # Train / test split
    train_val_index, test_index = train_test_split(
        np.arange(data_list_size),
        test_size=test_size,
        shuffle=True,
        stratify=labels,
        random_state=seed,
    )

    # train val split
    train_index, val_index = train_test_split(
        train_val_index,
        test_size=val_size / (train_size + val_size),
        shuffle=True,
        stratify=(labels[train_val_index] if labels is not None else None),
        random_state=seed,
    )
    return train_index, val_index, test_index
