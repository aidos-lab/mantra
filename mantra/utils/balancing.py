"""Dataset balancing via Pachner move augmentation."""

import bisect
import copy
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from mantra.manifold_types import Manifold2Type
from mantra.utils.constants import (
    BETTI_NUMBERS,
    CROSSCAP_GLUE_MAP,
    TORUS_GLUE_MAP,
)
from mantra.utils.deduplication import find_duplicates
from mantra.utils.triangulation import Triangulation

# Vertices of the glued torus / RP^2 minus the three identified ones.
GLUE_ADDS_N_VERTICES = {"torus": 4, "crosscap": 3}


def _genus_from_name(name):
    """Genus of a 2-manifold class, derived from its Betti numbers.

    Orientable genus g has b_1 = 2g; non-orientable genus k (number
    of crosscaps) has b_1 = k - 1.
    """
    betti = BETTI_NUMBERS[name]
    if betti[2] == 1:
        return betti[1] // 2
    return betti[1] + 1


def _augment_triangulation(
    entry: Dict,
    id_cnt: int,
    rng: random.Random,
    n_moves: int = 5,
    max_vertices: int | None = None,
):
    """Create a new triangulation by applying random Pachner moves.

    Parameters
    ----------
    entry : dict
        Dataset entry with 'triangulation' key.
    id_cnt : int
        Suffix of the new entry's id.
    rng : random.Random
        Random number generator.
    n_moves : int
        Number of random Pachner moves to apply.
    max_vertices : int or None
        If set, no move adds a vertex once the copy has this many.

    Returns
    -------
    dict
        New entry with modified triangulation and updated n_vertices.
    """
    new_entry = copy.deepcopy(entry)
    simplices = new_entry["triangulation"]

    t = Triangulation.from_list(simplices, rng=rng)

    for _ in range(n_moves):
        t.random_pachner_move(max_vertices=max_vertices)

    new_entry["triangulation"] = t.to_list()
    new_entry["n_vertices"] = t.n_vertices
    new_entry["id"] = f"{new_entry['id']}_aug_{id_cnt}"

    return new_entry


def _augment_with_surgery(
    entry: Dict, glue_type: str, id_cnt: int, rng: random.Random
):
    """Create a new triangulation by changing topology (2D only).

    Parameters
    ----------
    entry : Rict
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

    # Get the taret type from the entry
    target_manifold_class = (
        TORUS_GLUE_MAP[entry["name"]]
        if glue_type == "torus"
        else CROSSCAP_GLUE_MAP[entry["name"]]
    )

    new_entry = copy.deepcopy(entry)
    t = Triangulation.from_list(new_entry["triangulation"], rng=rng)
    t.glue(glue_type)
    new_entry["triangulation"] = t.to_list()
    new_entry["n_vertices"] = t.n_vertices
    new_entry["name"] = target_manifold_class
    new_entry["betti_numbers"] = list(BETTI_NUMBERS[target_manifold_class])
    new_entry["name"] = target_manifold_class
    new_entry["orientable"] = (
        glue_type == "torus"
    )  # Only torus gluing preserves orientability
    if "genus" in new_entry:
        new_entry["genus"] = _genus_from_name(target_manifold_class)

    #  Update the id to reflect source
    new_entry["id"] = f"{entry['id']}_glued_{glue_type}_{id_cnt}"

    return new_entry


def _find_topology_sources(
    target_manifold_class: str, class_entries: Dict[str, List]
):
    """Find classes that can produce the target via topology change.

    Returns
    -------
    list of tuple
        Source class names and glue types that can produce the target.
    """
    sources = []
    for name, entries in class_entries.items():
        # No entries for a given class name
        if len(entries) == 0:
            continue
        if TORUS_GLUE_MAP.get(name) == target_manifold_class:
            sources.append((name, "torus"))
        if CROSSCAP_GLUE_MAP.get(name) == target_manifold_class:
            sources.append((name, "crosscap"))
    return sources


def _deduplicate(class_entries, verbose=False, class_names: List = []):
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
    for manifold_name in class_names:
        entries = class_entries[manifold_name]

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


def do_surgery(
    class_entries: Dict[str, List],
    target_count: int,
    max_vertices: int | None,
    rng,
):
    """
    Perform surgical augmentations. So far this only works for 2D Manifolds.

    target_count : int
        Target count of samples per class.
    """
    id_cnt = 0
    augmented_classes = set()

    # Only go over the types we are missing
    missing_types = list(
        set([m_type.value for m_type in Manifold2Type])
        - set(list(class_entries.keys()))
    )

    # List of elements to apply surgery to
    to_surgery: List = []

    # For each type of manifold, we want to
    # see how we can construct it
    for target_manifold_name in missing_types:

        # Overshoot in case we get isomorphic triangulations
        deficit = target_count * 2

        # Return `source_manifold_names` that generate `manifold_name`
        # This are the manifolds that we can use to obtain our target
        # this contains a list of (name, glue_type) tuple
        source_manifold_names: List[Tuple[str, str]] = _find_topology_sources(
            target_manifold_name, class_entries
        )

        # Construct the manifolds we need to augment
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

                to_surgery.append((source_entry, glue_type))

                deficit -= 1

    # Performs the augmentations
    for source_entry, glue_type in to_surgery:
        new_entry = _augment_with_surgery(source_entry, glue_type, id_cnt, rng)
        target_manifold_name = new_entry["name"]

        # Add new entry
        bisect.insort(
            class_entries[target_manifold_name],
            new_entry,
            key=lambda x: x["n_vertices"],
        )

        augmented_classes.add(target_manifold_name)

        # Reduce counter
        id_cnt += 1

    return augmented_classes


def do_pachner(
    class_entries: Dict,
    target_count: int,
    max_vertices: int | None,
    n_moves: int,
    rng: random.Random,
):
    """Oversample every class to twice the target count by Pachner moves.

    Copies are drawn round-robin from the entries that are not
    themselves augmented copies and inserted in place, keeping each
    class sorted by ``n_vertices``. Copies exceeding ``max_vertices``
    are discarded and retried, up to ten attempts per missing entry.

    Returns
    -------
    set of str
        Names of the classes that gained augmented entries.
    """
    id_cnt = 0
    augmented_classes = set()

    for manifold_name, entries in class_entries.items():
        sources = [e for e in entries if "_aug_" not in e["id"]]
        missing = target_count * 2 - len(entries)
        if missing <= 0 or not sources:
            continue

        # Bounded so that an unsatisfiable vertex cap cannot loop forever.
        for attempt in range(10 * missing):
            if missing == 0:
                break
            new_entry = _augment_triangulation(
                sources[attempt % len(sources)],
                id_cnt,
                rng=rng,
                n_moves=n_moves,
                max_vertices=max_vertices,
            )
            if (
                max_vertices is not None
                and new_entry["n_vertices"] > max_vertices
            ):
                continue

            bisect.insort(entries, new_entry, key=lambda x: x["n_vertices"])
            augmented_classes.add(manifold_name)
            id_cnt += 1
            missing -= 1

    return augmented_classes


def balance_dataset(
    dataset: List,
    target_count: int,
    n_moves: int,
    use_surgery: bool,
    max_vertices: int | None,
    verbose: bool = False,
    seed: int = 42,
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
    dataset : list of Data
        List of torch Data objcets.
    target_count : int
        Target count per class.
    n_moves : int
        Number of Pachner moves per augmented sample.
    seed : int
        Random seed for reproducibility.
    use_surgery : bool
        If True and dimension==2, use surgery techniques
        (torus/crosscap gluing) to generate samples for missing
        classes that can be reached from existing ones.
    max_vertices : int or None
        If set, don't produce entries with more than this amount of vertices.
    verbose : bool
        If True, print progress to stderr.

    Returns
    -------
    list of dict
        Balanced dataset.
    """
    rng = random.Random(seed)

    dimension = len(dataset[0]["triangulation"][0]) - 1

    # Counts of each class
    class_entries = defaultdict(list)
    for entry in dataset:
        class_entries[entry["name"]].append(entry)

    # Sort the entries based on name (ascending)
    for manifold_name, entries in class_entries.items():
        class_entries[manifold_name] = sorted(
            entries, key=lambda x: x["n_vertices"], reverse=False
        )

    # Classes that gain augmented entries; only these need deduplication.
    augmented_classes = set()

    # In 2D we can do some glueings to generate more classes
    # which we are missing
    if dimension == 2 and use_surgery:
        new_agumented_classes = do_surgery(
            class_entries, target_count, max_vertices, rng
        )
        augmented_classes = augmented_classes.union(new_agumented_classes)

    # Perform pachner moves
    new_agumented_classes = do_pachner(
        class_entries,
        target_count=target_count,
        max_vertices=max_vertices,
        n_moves=n_moves,
        rng=rng,
    )
    augmented_classes = augmented_classes.union(new_agumented_classes)

    # Deduplicate the classes that gained augmented entries
    class_entries = _deduplicate(
        class_entries, verbose=verbose, class_names=list(augmented_classes)
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
