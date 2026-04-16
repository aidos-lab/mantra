"""Dataset balancing via Pachner move augmentation."""

import copy
import random
import sys
from collections import defaultdict

import numpy as np

from mantra.augmentations.triangulation_2d import Triangulation2D
from mantra.augmentations.triangulation_3d import Triangulation3D
from mantra.deduplication import find_duplicates

# Mapping from 2D manifold names to the name after gluing a torus.
# Based on the classification of closed surfaces:
# - Orientable genus g -> genus g+1
# - Non-orientable with k crosscaps + torus = k+2 crosscaps
_TORUS_GLUE_MAP = {
    "S^2": "T^2",
    "T^2": "#^2 T^2",
    "#^2 T^2": "#^3 T^2",
    "#^3 T^2": "#^4 T^2",
    "#^4 T^2": "#^5 T^2",
    "#^5 T^2": "#^6 T^2",
    "#^6 T^2": "#^7 T^2",
    # Non-orientable: torus + k crosscaps = k+2 crosscaps
    "RP^2": "#^3 RP^2",
    "#^2 RP^2": "#^4 RP^2",
    "#^3 RP^2": "#^5 RP^2",
    "#^4 RP^2": "#^6 RP^2",
    "#^5 RP^2": "#^7 RP^2",
    "#^6 RP^2": "#^8 RP^2",
}

# Mapping after gluing a crosscap (connected sum with RP^2).
# Orientable genus g + crosscap = 2g+1 crosscaps (non-orientable).
# Non-orientable k crosscaps + 1 = k+1 crosscaps.
_CROSSCAP_GLUE_MAP = {
    "S^2": "RP^2",
    "T^2": "#^3 RP^2",
    "RP^2": "#^2 RP^2",
    "#^2 RP^2": "#^3 RP^2",
    "#^2 T^2": "#^5 RP^2",
    "#^3 RP^2": "#^4 RP^2",
    "#^4 RP^2": "#^5 RP^2",
    "#^3 T^2": "#^7 RP^2",
    "#^5 RP^2": "#^6 RP^2",
    "#^6 RP^2": "#^7 RP^2",
}

# Betti numbers (over Z) for all 2-manifold classes.
# Orientable genus g: [1, 2g, 1]
# Non-orientable k crosscaps: [1, k-1, 0]
_BETTI_NUMBERS = {
    "S^2": [1, 0, 1],
    "T^2": [1, 2, 1],
    "#^2 T^2": [1, 4, 1],
    "#^3 T^2": [1, 6, 1],
    "#^4 T^2": [1, 8, 1],
    "#^5 T^2": [1, 10, 1],
    "#^6 T^2": [1, 12, 1],
    "#^7 T^2": [1, 14, 1],
    "RP^2": [1, 0, 0],
    "#^2 RP^2": [1, 1, 0],
    "#^3 RP^2": [1, 2, 0],
    "#^4 RP^2": [1, 3, 0],
    "#^5 RP^2": [1, 4, 0],
    "#^6 RP^2": [1, 5, 0],
    "#^7 RP^2": [1, 6, 0],
    "#^8 RP^2": [1, 7, 0],
}


def _augment_triangulation(entry, dimension, n_moves=5, rng=None):
    """Create a new triangulation by applying random Pachner moves.

    Parameters
    ----------
    entry : dict
        Dataset entry with 'triangulation' key.
    dimension : int
        2 or 3.
    n_moves : int
        Number of random Pachner moves to apply.
    rng : random.Random or None
        Random number generator.

    Returns
    -------
    dict
        New entry with modified triangulation and updated n_vertices.
    """
    new_entry = copy.deepcopy(entry)
    simplices = new_entry["triangulation"]

    if dimension == 2:
        t = Triangulation2D(simplices, rng=rng)
        for _ in range(n_moves):
            t.random_pachner_move()
    else:
        t = Triangulation3D(simplices, rng=rng)
        for _ in range(n_moves):
            t.random_pachner_move()

    new_entry["triangulation"] = t.to_list()
    new_entry["n_vertices"] = t.n_vertices
    return new_entry


def _augment_with_topology_change(entry, target_name, rng=None):
    """Create a new triangulation by changing topology (2D only).

    Parameters
    ----------
    entry : dict
        Source entry.
    target_name : str
        Target manifold class name.
    rng : random.Random or None
        Random number generator.

    Returns
    -------
    dict or None
        New entry with changed topology, or None if not possible.
    """
    source_name = entry["name"]

    # try torus gluing
    if _TORUS_GLUE_MAP.get(source_name) == target_name:
        new_entry = copy.deepcopy(entry)
        t = Triangulation2D(new_entry["triangulation"], rng=rng)
        t.glue_torus()
        new_entry["triangulation"] = t.to_list()
        new_entry["n_vertices"] = t.n_vertices
        new_entry["name"] = target_name
        new_entry["betti_numbers"] = list(_BETTI_NUMBERS[target_name])
        if "genus" in new_entry:
            new_entry["genus"] = new_entry.get("genus", 0) + 1
        return new_entry

    # try crosscap gluing
    if _CROSSCAP_GLUE_MAP.get(source_name) == target_name:
        new_entry = copy.deepcopy(entry)
        t = Triangulation2D(new_entry["triangulation"], rng=rng)
        t.glue_crosscap()
        new_entry["triangulation"] = t.to_list()
        new_entry["n_vertices"] = t.n_vertices
        new_entry["name"] = target_name
        new_entry["betti_numbers"] = list(_BETTI_NUMBERS[target_name])
        new_entry["orientable"] = False
        return new_entry

    return None


def _find_topology_sources(target_name, class_entries):
    """Find classes that can produce the target via topology change.

    Returns
    -------
    list of str
        Source class names that can produce the target.
    """
    sources = []
    for name, entries in class_entries.items():
        if not entries:
            continue
        if _TORUS_GLUE_MAP.get(name) == target_name:
            sources.append(name)
        if _CROSSCAP_GLUE_MAP.get(name) == target_name:
            sources.append(name)
    return sources


def _downsample(entries, target_count, rng):
    """Downsample entries preserving vertex count distribution.

    Parameters
    ----------
    entries : list of dict
        Entries to downsample.
    target_count : int
        Target number of entries.
    rng : random.Random
        Random number generator.

    Returns
    -------
    list of dict
        Downsampled entries.
    """
    if len(entries) <= target_count:
        return entries

    # stratified sampling by vertex count
    by_nverts = defaultdict(list)
    for e in entries:
        by_nverts[e["n_vertices"]].append(e)

    total = len(entries)
    result = []
    for nv in sorted(by_nverts.keys()):
        group = by_nverts[nv]
        # proportional allocation
        n_select = max(1, round(len(group) / total * target_count))
        n_select = min(n_select, len(group))
        result.extend(rng.sample(group, n_select))

    # adjust to exact target count
    if len(result) > target_count:
        result = rng.sample(result, target_count)
    elif len(result) < target_count:
        result_ids = {id(e) for e in result}
        remaining = [e for e in entries if id(e) not in result_ids]
        needed = target_count - len(result)
        result.extend(rng.sample(remaining, min(needed, len(remaining))))

    return result


def balance_dataset(
    dataset,
    dimension,
    target_count=1000,
    n_moves=5,
    seed=42,
    use_topology_changes=True,
    dedup_max_rounds=10,
    max_vertices=None,
    verbose=False,
):
    """Generate a balanced dataset via Pachner move augmentation.

    After augmentation, runs deduplication to remove isomorphic
    duplicates (e.g. a Pachner-moved copy that happens to be
    isomorphic to an existing triangulation). Removed duplicates
    are replaced with fresh augmentations and re-checked, up to
    ``dedup_max_rounds`` times.

    Parameters
    ----------
    dataset : list of dict
        Raw JSON entries with 'triangulation', 'name', etc.
    dimension : int
        2 or 3.
    target_count : int
        Target count per class.
    n_moves : int
        Number of Pachner moves per augmented sample.
    seed : int
        Random seed for reproducibility.
    use_topology_changes : bool
        If True and dimension==2, use topology-changing operations
        to generate samples for classes that can be reached.
    dedup_max_rounds : int
        Maximum number of dedup-regenerate rounds. Set to 0 to
        skip deduplication entirely.
    max_vertices : int or None
        If set, discard all entries (original and augmented) with
        more than this many vertices. Applied both before balancing
        and as a final filter after augmentation and deduplication.
    verbose : bool
        If True, print progress to stderr.

    Returns
    -------
    list of dict
        Balanced dataset.
    """
    rng = random.Random(seed)

    # filter by max_vertices before balancing
    if max_vertices is not None:
        dataset = [
            e for e in dataset if e["n_vertices"] <= max_vertices
        ]

    # group by class
    class_entries = defaultdict(list)
    for entry in dataset:
        class_entries[entry["name"]].append(entry)

    result = []
    aug_counter = defaultdict(int)

    for name, entries in class_entries.items():
        if len(entries) >= target_count:
            # downsample
            result.extend(_downsample(entries, target_count, rng))
        else:
            # keep all originals
            result.extend(entries)
            # oversample with Pachner moves
            deficit = target_count - len(entries)
            for _ in range(deficit):
                seed_entry = rng.choice(entries)
                new_entry = _augment_triangulation(
                    seed_entry, dimension, n_moves, rng=rng
                )
                aug_counter[name] += 1
                new_entry["id"] = (
                    f"{seed_entry['id']}" f"_aug_{aug_counter[name]}"
                )
                result.append(new_entry)

    # topology-changing augmentation for 2D
    if dimension == 2 and use_topology_changes:
        current_counts = defaultdict(int)
        for e in result:
            current_counts[e["name"]] += 1

        # find all names that could exist via topology changes
        all_possible_names = set(current_counts.keys())
        all_possible_names |= set(_TORUS_GLUE_MAP.values())
        all_possible_names |= set(_CROSSCAP_GLUE_MAP.values())

        for target_name in all_possible_names:
            current = current_counts.get(target_name, 0)
            if current >= target_count:
                continue

            sources = _find_topology_sources(target_name, class_entries)
            if not sources:
                continue

            deficit = target_count - current
            for _ in range(deficit):
                source_name = rng.choice(sources)
                source_entry = rng.choice(class_entries[source_name])
                new_entry = _augment_with_topology_change(
                    source_entry, target_name, rng=rng
                )
                if new_entry is None:
                    continue
                aug_counter[target_name] += 1
                new_entry["id"] = (
                    f"{source_entry['id']}" f"_topo_{aug_counter[target_name]}"
                )
                # also apply some Pachner moves for diversity
                augmented = _augment_triangulation(
                    new_entry, dimension, n_moves, rng=rng
                )
                augmented["id"] = new_entry["id"]
                result.append(augmented)

    # Post-augmentation cleanup: remove isomorphic duplicates and
    # entries exceeding max_vertices, then regenerate replacements.
    # On the final round, only remove without regenerating —
    # prioritising a clean dataset over hitting the exact target
    # count.
    for dedup_round in range(dedup_max_rounds):
        is_last_round = dedup_round == dedup_max_rounds - 1

        # find duplicates
        duplicates = find_duplicates(result, verbose=verbose)
        dup_ids = {pair[1] for pair in duplicates}

        # find entries exceeding vertex limit
        over_limit_ids = set()
        if max_vertices is not None:
            over_limit_ids = {
                e["id"]
                for e in result
                if e["n_vertices"] > max_vertices
            }

        to_remove = dup_ids | over_limit_ids
        if not to_remove:
            if verbose:
                print(
                    f"Dedup round {dedup_round + 1}: no duplicates "
                    f"or vertex violations found.",
                    file=sys.stderr,
                )
            break

        if verbose:
            action = "removing only" if is_last_round else "removing"
            parts = []
            if dup_ids:
                parts.append(f"{len(dup_ids)} duplicates")
            if over_limit_ids:
                parts.append(
                    f"{len(over_limit_ids)} entries exceeding "
                    f"max_vertices={max_vertices}"
                )
            print(
                f"Dedup round {dedup_round + 1}: {action} "
                f"{' and '.join(parts)}.",
                file=sys.stderr,
            )

        # group removed entries by class for targeted regeneration
        removed_by_class = defaultdict(int)
        kept = []
        for e in result:
            if e["id"] in to_remove:
                removed_by_class[e["name"]] += 1
            else:
                kept.append(e)
        result = kept

        # On the last round, skip regeneration to guarantee no
        # new duplicates are introduced.
        if is_last_round:
            break

        # regenerate replacements for each class
        for name, n_removed in removed_by_class.items():
            originals = class_entries.get(name, [])
            if not originals:
                continue
            for _ in range(n_removed):
                seed_entry = rng.choice(originals)
                new_entry = _augment_triangulation(
                    seed_entry, dimension, n_moves, rng=rng
                )
                aug_counter[name] += 1
                new_entry["id"] = (
                    f"{seed_entry['id']}" f"_aug_{aug_counter[name]}"
                )
                result.append(new_entry)

    # Safety net: if dedup_max_rounds is 0 (loop skipped entirely),
    # still enforce the vertex limit.
    if max_vertices is not None and dedup_max_rounds == 0:
        result = [
            e for e in result if e["n_vertices"] <= max_vertices
        ]

    return result


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
            f"{name:<30} {len(entries):>8}"
            f"{min(nverts):>8} {round(np.mean(nverts),2):>8} {max(nverts):>8}"
        )
    print(f"\nTotal: {len(dataset)}")
