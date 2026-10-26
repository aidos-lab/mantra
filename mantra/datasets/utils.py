import multiprocessing as mp
import os
import pickle
import random
from collections import Counter
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

_worker_transform = None


def _init_worker(transform: Callable) -> None:
    global _worker_transform
    _worker_transform = transform
    torch.set_num_threads(1)


def _apply_worker_transform(seed_and_data: Tuple[int, bytes]) -> bytes:
    # Plain pickle bypasses torch's shared-memory reductions, which keep
    # one file descriptor per storage in the parent and exhaust `ulimit -n`.
    seed, pickled_data = seed_and_data
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    data = _worker_transform(pickle.loads(pickled_data))
    return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def available_cpus() -> int:
    """Number of CPUs this process may run on.

    Uses the affinity mask where the OS exposes it, so a SLURM or cgroup
    allocation is respected.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def map_pre_transform(
    pre_transform: Callable,
    data_list: List,
    num_workers: Optional[int] = None,
) -> List:
    """Apply `pre_transform` to every element of `data_list`, in order.

    In the pool, every sample is transformed under its own seed derived
    from `torch.initial_seed()` without touching the global generator,
    so transforms using the global torch, numpy or `random` generators
    give the same result for any worker count and schedule. Forked
    workers share the state of a generator a transform holds itself, so
    such transforms must run with `num_workers=1`.

    Parameters
    ----------
    num_workers : int, optional
        Worker processes; `None` uses every available CPU and `1` runs
        in this process. The transform must be picklable when the
        start method is not fork.
    """
    if num_workers is None:
        num_workers = available_cpus()
    num_workers = min(num_workers, len(data_list) // 2)
    if num_workers <= 1:
        return [
            pre_transform(data)
            for data in tqdm(data_list, desc="Pre-transforming")
        ]

    methods = mp.get_all_start_methods()
    ctx = mp.get_context("fork" if "fork" in methods else methods[0])
    chunksize = max(1, min(64, len(data_list) // (4 * num_workers)))
    generator = torch.Generator().manual_seed(torch.initial_seed())
    seeds = torch.randint(
        0, 2**32 - 1, (len(data_list),), generator=generator
    ).tolist()
    jobs = (
        (seed, pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        for seed, data in zip(seeds, data_list)
    )
    with ctx.Pool(
        num_workers, initializer=_init_worker, initargs=(pre_transform,)
    ) as pool:
        return [
            pickle.loads(result)
            for result in tqdm(
                pool.imap(_apply_worker_transform, jobs, chunksize),
                total=len(data_list),
                desc="Pre-transforming",
            )
        ]


def _get_mantra_dataset_url(version: str, dimension: int) -> str:
    """Get URL to download dataset from."""
    filename = f"{dimension}_manifolds.json.gz"

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
