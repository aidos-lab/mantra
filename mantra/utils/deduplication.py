"""Deduplication utilities for manifold triangulations.

Detects duplicate triangulations that are combinatorially isomorphic
(same simplicial complex up to vertex relabeling). Uses a multi-level
filtering strategy for efficiency:

1. Cheap invariants (f-vector, degree sequence) to group triangulations
2. WL graph hash of the incidence graph to create subgroups
3. Full simplicial complex isomorphism checking within subgroups

The key insight is using the **incidence graph** (bipartite graph between
vertex nodes and top-simplex nodes) rather than the 1-skeleton. Many
3-manifold triangulations have the complete graph K_n as 1-skeleton,
making 1-skeleton-based isomorphism extremely slow (n! automorphisms).
The incidence graph encodes the full simplicial complex structure and
has far less symmetry.

Usage as a script::

    python -m mantra.deduplication path/to/manifolds.json
"""

import json
import sys
from collections import defaultdict

from mantra.utils.utils import (
    are_isomorphic,
    compute_invariant_key,
    compute_wl_hash,
)


def find_duplicates(triangulations, verbose=False, iso_max_group_size=5):
    """Find duplicate triangulations in a dataset.

    Uses a three-level filtering strategy:
    1. Group by cheap invariants (f-vector + degree sequence)
    2. Subgroup by WL hash of the incidence graph
    3. Pairwise simplicial complex isomorphism within subgroups

    For WL subgroups larger than ``iso_max_group_size``, the
    expensive pairwise VF2 isomorphism check is skipped.
    Instead, all but one member are treated as duplicates based
    on the WL hash alone — a conservative but fast strategy.

    Parameters
    ----------
    triangulations : list of dict
        Each dict must have ``'triangulation'`` and ``'id'`` keys.
    verbose : bool
        If True, print progress to stderr.
    iso_max_group_size : int
        Maximum WL subgroup size for which full pairwise
        isomorphism checks are performed. Larger subgroups
        use WL-hash-only deduplication. Set to 0 to always
        skip isomorphism checks.

    Returns
    -------
    list of tuple
        List of ``(id1, id2)`` pairs that are isomorphic duplicates.
    """
    n = len(triangulations)

    # Level 1: Group by cheap invariants
    if verbose:
        print(
            f"Computing invariants for {n} triangulations...",
            file=sys.stderr,
        )

    invariant_groups = defaultdict(list)
    for i, tri_data in enumerate(triangulations):
        tri = tri_data["triangulation"]
        key = compute_invariant_key(tri)
        invariant_groups[key].append(tri_data)
        if verbose and (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{n}", file=sys.stderr)

    nontrivial = {k: v for k, v in invariant_groups.items() if len(v) > 1}
    if verbose:
        print(
            f"  {len(invariant_groups)} invariant groups, "
            f"{len(nontrivial)} with >1 member",
            file=sys.stderr,
        )

    # Level 2 & 3: Within each group, subgroup by WL hash, then check
    duplicates = []
    total_groups = len(nontrivial)

    for group_idx, (key, members) in enumerate(nontrivial.items(), 1):
        if verbose:
            print(
                f"Group {group_idx}/{total_groups}: "
                f"f={key[0]}, deg_seq_len={len(key[1])}, "
                f"size={len(members)}",
                file=sys.stderr,
            )

        # Subgroup by WL hash of incidence graph
        wl_groups = defaultdict(list)
        for tri_data in members:
            wl = compute_wl_hash(tri_data["triangulation"])
            wl_groups[wl].append(tri_data)

        if verbose:
            nontrivial_wl = sum(1 for v in wl_groups.values() if len(v) > 1)
            max_wl = max(len(v) for v in wl_groups.values())
            print(
                f"  -> {len(wl_groups)} WL subgroups, "
                f"{nontrivial_wl} nontrivial, max size={max_wl}",
                file=sys.stderr,
            )

        # Pairwise isomorphism check within WL subgroups
        for wl_members in wl_groups.values():
            if len(wl_members) < 2:
                continue

            if len(wl_members) > iso_max_group_size:
                # Group too large for pairwise VF2 — treat WL
                # hash collisions as duplicates (keep first only).
                if verbose:
                    print(
                        f"  WL subgroup size {len(wl_members)} > "
                        f"{iso_max_group_size}: skipping VF2, "
                        f"deduplicating by WL hash",
                        file=sys.stderr,
                    )
                for j in range(1, len(wl_members)):
                    duplicates.append(
                        (wl_members[0]["id"], wl_members[j]["id"])
                    )
                continue

            for i in range(len(wl_members)):
                for j in range(i + 1, len(wl_members)):
                    if are_isomorphic(
                        wl_members[i]["triangulation"],
                        wl_members[j]["triangulation"],
                    ):
                        duplicates.append(
                            (wl_members[i]["id"], wl_members[j]["id"])
                        )

    return duplicates


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m mantra.deduplication <path_to_json>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading {path}...", file=sys.stderr)

    with open(path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} triangulations.", file=sys.stderr)

    duplicates = find_duplicates(data, verbose=True)

    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate pairs:", file=sys.stderr)
        for id1, id2 in duplicates:
            print(f"  {id1} <-> {id2}")
        sys.exit(1)
    else:
        print("\nNo duplicates found.", file=sys.stderr)
        sys.exit(0)
