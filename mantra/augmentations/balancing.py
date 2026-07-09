"""Dataset balancing via Pachner move augmentation."""

import bisect
import copy
import random
import sys
from collections import defaultdict
from typing import List, Tuple

from mantra.augmentations.constants import (
    BETTI_NUMBERS,
    CROSSCAP_GLUE_MAP,
    TORUS_GLUE_MAP,
)
from mantra.augmentations.triangulation import Triangulation
from mantra.manifold_types import Manifold2Type
from mantra.utils.deduplication import find_duplicates


def _genus_from_name(name):
    """Genus of a 2-manifold class, derived from its Betti numbers.

    Orientable genus g has b_1 = 2g; non-orientable genus k (number
    of crosscaps) has b_1 = k - 1.
    """
    betti = BETTI_NUMBERS[name]
    if betti[2] == 1:
        return betti[1] // 2
    return betti[1] + 1


def _augment_triangulation(entry, n_moves=5, rng=None):
    """Create a new triangulation by applying random Pachner moves.

    Parameters
    ----------
    entry : dict
        Dataset entry with 'triangulation' key.
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

    t = Triangulation.from_list(simplices, rng=rng)

    for _ in range(n_moves):
        t.random_pachner_move()

    new_entry["triangulation"] = t.to_list()
    new_entry["n_vertices"] = t.n_vertices
    return new_entry


def _augment_with_topology_change(entry, glue_type, rng=None):
    """Create a new triangulation by changing topology (2D only).

    Parameters
    ----------
    entry : dict
        Source entry.
    glue_type : str
        Type of topology change: 'torus' or 'crosscap'.
    rng : random.Random or None
        Random number generator.

    Returns
    -------
    dict or None
        New entry with changed topology, or None if not possible.
    """
    target_manifold_class = (
        TORUS_GLUE_MAP.get(entry["name"])
        if glue_type == "torus"
        else CROSSCAP_GLUE_MAP.get(entry["name"])
    )

    new_entry = copy.deepcopy(entry)
    t = Triangulation.from_list(new_entry["triangulation"], rng=rng)
    t.glue(glue_type)
    new_entry["triangulation"] = t.to_list()
    new_entry["n_vertices"] = t.n_vertices
    new_entry["name"] = target_manifold_class
    new_entry["betti_numbers"] = list(BETTI_NUMBERS[target_manifold_class])
    new_entry["orientable"] = (
        glue_type == "torus"
    )  # Only torus gluing preserves orientability
    if "genus" in new_entry:
        new_entry["genus"] = _genus_from_name(target_manifold_class)

    return new_entry


def _find_topology_sources(target_manifold_class, class_entries):
    """Find classes that can produce the target via topology change.

    Returns
    -------
    list of tuple
        Source class names and glue types that can produce the target.
    """
    sources = []
    for name, entries in class_entries.items():
        if not entries:
            continue
        if TORUS_GLUE_MAP.get(name) == target_manifold_class:
            sources.append((name, "torus"))
        if CROSSCAP_GLUE_MAP.get(name) == target_manifold_class:
            sources.append((name, "crosscap"))
    return sources


def _deduplicate(class_entries, verbose=False, class_names=None):
    """Remove isomorphic duplicates from each class (single pass).

    For every duplicate pair reported by ``find_duplicates``, the
    second entry is dropped. Entries are not regenerated; callers
    oversample beforehand so that enough entries survive.

    Parameters
    ----------
    class_names : set of str or None
        If given, only these classes are scanned. Classes that received
        no augmented entries contain only upstream-curated originals,
        so scanning them wastes the bulk of the deduplication cost.
    """
    for manifold_name, entries in class_entries.items():
        if class_names is not None and manifold_name not in class_names:
            continue
        duplicates = find_duplicates(entries, verbose=verbose)

        # Get the id of the second duplicate
        dup_ids = {pair[1] for pair in duplicates}

        if verbose:
            message = (
                f"Dedup: removing {len(dup_ids)} duplicates "
                f"from class {manifold_name}."
                if dup_ids
                else f"Dedup: no duplicates found in class {manifold_name}."
            )
            print(message, file=sys.stderr)

        class_entries[manifold_name] = [
            e for e in entries if e["id"] not in dup_ids
        ]
    return class_entries


def balance_dataset(
    dataset,
    target_count=1000,
    n_moves=5,
    seed=42,
    use_topology_changes=True,
    max_vertices=None,
    verbose=False,
):
    """Generate a balanced dataset via Pachner move augmentation.

    Each class is oversampled to twice the target count, isomorphic
    duplicates (e.g. a Pachner-moved copy that happens to be
    isomorphic to an existing triangulation) are removed in a single
    deduplication pass over the classes that gained augmented entries,
    and a random subsample of ``target_count`` entries per class is
    kept. The dimension is derived from the data.

    Parameters
    ----------
    dataset : list of dict
        Raw JSON entries with 'triangulation', 'name', etc. The
        returned list shares entry dicts with the input.
    target_count : int
        Target count per class.
    n_moves : int
        Number of Pachner moves per augmented sample.
    seed : int
        Random seed for reproducibility.
    use_topology_changes : bool
        If True and dimension==2, use topology-changing operations
        (torus/crosscap gluing) to generate samples for missing
        classes that can be reached from existing ones.
    max_vertices : int or None
        If set, discard all entries with more than this many vertices
        before balancing.
    verbose : bool
        If True, print progress to stderr.

    Returns
    -------
    list of dict
        Balanced dataset.
    """
    rng = random.Random(seed)
    dimension = len(dataset[0]["triangulation"][0]) - 1

    # filter by max_vertices before balancing
    if max_vertices is not None:
        dataset = [e for e in dataset if e["n_vertices"] <= max_vertices]

    # Counts of each class
    class_entries = defaultdict(list)
    for entry in dataset:
        class_entries[entry["name"]].append(entry)

    # Sort the entries based on name (ascending)
    for manifold_name, entries in class_entries.items():
        class_entries[manifold_name] = sorted(
            entries, key=lambda x: x["n_vertices"], reverse=False
        )

    GLUE_ADDS_N_VERTICES = {"torus": 3, "crosscap": 1}
    id_cnt = 0
    # Classes that gain augmented entries; only these need deduplication.
    augmented_classes = set()

    # In 2D we can do some glueings to generate more classes
    # which we are missing
    if dimension == 2 and use_topology_changes:
        for obj_manifold in Manifold2Type:
            # Set name
            target_manifold_name = obj_manifold.value

            # We already have this manifold
            if target_manifold_name in class_entries:
                continue
            deficit = target_count * 2
            # Return `source_manifold_names` that generate `manifold_name`
            source_manifold_names: List[Tuple[str, str]] = (
                _find_topology_sources(target_manifold_name, class_entries)
            )

            # class_entries[x[0]] (manifold type) is the list of triangulations
            # we grab the manifold type with the minimal triangulation
            # since they are sorted, we grab the first one
            source_manifold_names = sorted(
                source_manifold_names,
                key=lambda x: class_entries[x[0]][0]["n_vertices"],
                reverse=False,
            )

            for source_manifold_name, glue_type in source_manifold_names:
                amount = min(deficit, len(class_entries[source_manifold_name]))
                # Try as many as we can fit
                for i in range(amount):
                    source_entry = class_entries[source_manifold_name][i]

                    # Glueing always add new vertices
                    if (
                        max_vertices is not None
                        and source_entry["n_vertices"]
                        + GLUE_ADDS_N_VERTICES[glue_type]
                        > max_vertices
                    ):
                        break

                    # Perform the glueing
                    new_entry = _augment_with_topology_change(
                        source_entry, glue_type=glue_type, rng=rng
                    )

                    #  Update the id to reflect source
                    new_entry["id"] = (
                        f"{source_entry['id']}_glued_{glue_type}_{id_cnt}"
                    )

                    # Add new entry
                    bisect.insort(
                        class_entries[target_manifold_name],
                        new_entry,
                        key=lambda x: x["n_vertices"],
                    )
                    augmented_classes.add(target_manifold_name)

                    # Reduce counter
                    deficit -= 1
                    id_cnt += 1

    # For each name (manifold class) and a list of all entries (triangulation)
    # of that class
    for manifold_name, entries in class_entries.items():
        # If we have more than enough entries
        if len(entries) >= target_count * 2:
            continue

        # Oversample with Pachner moves, cycling through a snapshot of
        # the current entries: augmented copies must not become sources
        # of further augmentation, or minority classes drift arbitrarily
        # far from real data. Each move adds at most one vertex, so
        # sources within max_vertices - n_moves stay under the limit.
        sources = [
            e
            for e in entries
            if max_vertices is None
            or e["n_vertices"] + n_moves <= max_vertices
        ]
        if not sources:
            raise ValueError(
                f"Cannot balance class '{manifold_name}': all "
                f"{len(entries)} of its triangulations have more than "
                f"max_vertices - n_moves = {max_vertices - n_moves} "
                "vertices, so no augmented copy can stay under the "
                "vertex limit. Increase max_vertices, reduce n_moves, "
                "or filter the class out before balancing."
            )

        deficit = target_count * 2 - len(entries)
        for i in range(deficit):
            source_entry = sources[i % len(sources)]

            # Augment
            new_entry = _augment_triangulation(source_entry, n_moves, rng=rng)

            new_entry["id"] = f"{source_entry['id']}_aug_{id_cnt}"

            # Sorted insert
            bisect.insort(
                class_entries[manifold_name],
                new_entry,
                key=lambda x: x["n_vertices"],
            )
            augmented_classes.add(manifold_name)

            id_cnt += 1

    # Deduplicate the classes that gained augmented entries
    class_entries = _deduplicate(
        class_entries, verbose=verbose, class_names=augmented_classes
    )

    for manifold_name, entries in class_entries.items():
        if len(entries) < target_count:
            raise ValueError(
                f"Deduplication left class '{manifold_name}' with "
                f"{len(entries)} < target_count ({target_count}) "
                "entries: the augmented copies were too similar to "
                "survive. Increase n_moves or lower target_count."
            )
        if len(entries) > target_count:
            # Random subsample down to the target: slicing the
            # vertex-sorted list would keep only the smallest
            # triangulations and bias the vertex distribution.
            class_entries[manifold_name] = rng.sample(entries, target_count)

    # Collect the balanced classes
    resulting_entries = []
    for manifold_name, entries in class_entries.items():
        resulting_entries.extend(entries)

    return resulting_entries
