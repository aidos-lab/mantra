"""Generate subdivided triangulation datasets from a raw MANTRA JSON file.

Applies repeated subdivision to every entry in a MANTRA-style raw triangulation
JSON file and writes one output JSON per subdivision level. Entries whose
subdivided vertex count is below ``min_vertices`` are dropped from the output
(but kept in the running list so later levels are computed from the full
intermediate).

Three subdivision modes are supported:

* ``full_barycentric`` -- the classical barycentric subdivision (the order
  complex); each top simplex becomes ``d!`` simplices.
* ``stellar`` -- stellar subdivision; supports a *fractional* level in (0, 1)
  that subdivides only a random fraction of the top simplices.
* ``graded`` -- repeated stellar subdivision until a target vertex count is
  reached (``n_levels`` is interpreted as that target).

Optionally applies a pre-subdivision class-count filter (``--min-class-count``)
so that downstream training and any precomputed caches see the same class set.

The geometry is delegated to the subdivision moves on
:class:`mantra.augmentations.base.Triangulation` via :mod:`mantra.subdivision`.

Example
-------
    python -m mantra.generate_subdivided_datasets \\
        --input data/2D/raw/2_manifolds.json \\
        --output-dir /tmp/mantra_bary \\
        --prefix 2_manifolds \\
        --mode full_barycentric \\
        --n-levels 4 \\
        --min-vertices 16 \\
        --min-class-count 100
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

from mantra.subdivision import (
    barycentric_stellar_graded,
    barycentric_subdivision_raw,
    stellar_subdivision_raw,
)

_SUBDIVISION_FUNCTIONS = {
    "stellar": stellar_subdivision_raw,
    "full_barycentric": barycentric_subdivision_raw,
    "graded": barycentric_stellar_graded,
}


def _subdivide_entry(entry, mode, fraction=1.0, rng=None):
    """Subdivide a single dataset entry, returning a new entry."""
    fn = _SUBDIVISION_FUNCTIONS[mode]
    if mode == "stellar":
        tri, n_v = fn(entry["triangulation"], fraction=fraction, rng=rng)
    elif mode == "graded":
        tri, n_v = fn(entry["triangulation"], over_vrtx_cnt=fraction, rng=rng)
    else:
        tri, n_v = fn(entry["triangulation"])
    new_entry = dict(entry)
    new_entry["triangulation"] = tri
    new_entry["n_vertices"] = n_v
    return new_entry


def subdivide_once(entries, mode, fraction=1.0, rng=None):
    """Subdivide every entry once."""
    return [
        _subdivide_entry(e, mode, fraction=fraction, rng=rng) for e in entries
    ]


def _level_tag(level):
    """Format a level for the output key: integers as '1', fractions as '0.5'."""
    return f"{level:g}"


def filter_by_class_count(entries, label_source, min_count):
    """Drop entries whose ``label_source`` value occurs <= ``min_count`` times."""
    if min_count <= 0:
        return entries, Counter()
    counts = Counter(e[label_source] for e in entries)
    kept_labels = {lbl for lbl, c in counts.items() if c > min_count}
    filtered = [e for e in entries if e[label_source] in kept_labels]
    return filtered, counts


def print_statistics(dataset, label_source="name"):
    """Print per-class n_vertices statistics for a list of entries."""
    class_entries = defaultdict(list)
    for entry in dataset:
        class_entries[entry[label_source]].append(entry)

    print(
        f"{'Class':<30} {'Count':>8} {'Min V':>8} {'Mean V':>8} {'Max V':>8}"
    )
    print("-" * 64)
    for name in sorted(class_entries.keys()):
        entries = class_entries[name]
        nverts = [e["n_vertices"] for e in entries]
        mean_v = sum(nverts) / len(nverts)
        print(
            f"{str(name):<30} {len(entries):>8} "
            f"{min(nverts):>8} {mean_v:>8.2f} {max(nverts):>8}"
        )
    print(f"\nTotal: {len(dataset)}")


def _resolve_levels(mode, n_levels):
    """Validate ``n_levels`` for ``mode`` and resolve the level parameters.

    Returns
    -------
    fractional : bool
        Whether a single partial (fraction < 1) stellar pass is requested.
    n_full_levels : int
        Number of full integer subdivision rounds.
    fraction : float
        The per-pass fraction (stellar) or vertex target (graded). For
        ``mode='graded'`` this is the integer-valued vertex target.
    """
    nl = float(n_levels)

    # Graded mode interprets n_levels as a vertex target, not a fraction or a
    # round count, so it has its own validation: a positive integer.
    if mode == "graded":
        target = int(round(nl))
        if abs(nl - target) > 1e-9 or target <= 0:
            raise ValueError(
                f"graded target (n_levels) must be a positive integer "
                f"vertex count, got {nl}."
            )
        return False, 0, target

    if nl <= 0:
        raise ValueError(f"n_levels must be > 0, got {nl}")

    fractional = nl < 1.0
    if fractional and mode != "stellar":
        raise ValueError(
            "Fractional n_levels (partial subdivision) only supported in "
            "mode='stellar'; full_barycentric breaks the SC condition under "
            "partial application."
        )

    n_full_levels = 0 if fractional else int(round(nl))
    if not fractional and abs(nl - n_full_levels) > 1e-9:
        raise ValueError(
            f"n_levels >= 1 must be integer-valued (got {nl}); fractional "
            f"values are only allowed in (0, 1)."
        )

    fraction = nl if fractional else 1.0
    return fractional, n_full_levels, fraction


def generate_levels(
    data,
    *,
    mode="full_barycentric",
    n_levels=1,
    min_vertices=0,
    label_source="name",
    min_class_count=0,
    n_smallest=0,
    seed=42,
):
    """Produce subdivided datasets keyed by output level.

    Returns
    -------
    dict[str, list]
        Mapping from an output key (e.g. ``"bary_1"`` or ``"graded_n12"``) to
        the list of subdivided entries for that level, in generation order.
    """
    if mode not in _SUBDIVISION_FUNCTIONS:
        raise ValueError(
            f"Unknown subdivision mode {mode!r}; expected one of "
            f"{sorted(_SUBDIVISION_FUNCTIONS)}"
        )

    fractional, n_full_levels, fraction = _resolve_levels(mode, n_levels)

    if min_class_count > 0:
        data, _ = filter_by_class_count(data, label_source, min_class_count)

    rng = random.Random(seed)
    first_level_tag = _level_tag(fraction if fractional else 1)
    outputs = {}

    if n_smallest > 0:
        cohort = _select_smallest_cohort(
            data,
            mode=mode,
            fraction=fraction,
            min_vertices=min_vertices,
            label_source=label_source,
            n_smallest=n_smallest,
            rng=rng,
        )
        if mode == "graded":
            outputs[f"graded_n{fraction}"] = cohort
        else:
            outputs[f"bary_{first_level_tag}"] = cohort

        if fractional:
            return outputs

        current = cohort
        for level in range(2, n_full_levels + 1):
            current = subdivide_once(current, mode)
            below = [e for e in current if e["n_vertices"] < min_vertices]
            # Defensive: subdivision is monotonic in vertex count, so a
            # cohort entry that already cleared ``min_vertices`` cannot drop
            # below it on a further level. Kept as a guard.
            if below:  # pragma: no cover
                print(
                    f"[WARN] Level {level}: {len(below)} entries below "
                    f"{min_vertices} vertices after subdivision (unexpected)."
                )
            outputs[f"bary_{_level_tag(level)}"] = current
        return outputs

    if fractional:
        current = subdivide_once(data, mode, fraction=fraction, rng=rng)
        filtered = [e for e in current if e["n_vertices"] >= min_vertices]
        outputs[f"bary_{first_level_tag}"] = filtered
        return outputs

    current = data
    for level in range(1, n_full_levels + 1):
        current = subdivide_once(current, mode)
        filtered = [e for e in current if e["n_vertices"] >= min_vertices]
        outputs[f"bary_{_level_tag(level)}"] = filtered
    return outputs


def _select_smallest_cohort(
    data, *, mode, fraction, min_vertices, label_source, n_smallest, rng
):
    """Subdivide the ``n_smallest`` valid entries per class.

    Iterates each class's entries (ascending by vertex count, descending for
    graded), subdividing until ``n_smallest`` results clear ``min_vertices``.
    """
    per_class = defaultdict(list)
    for entry in data:
        per_class[entry[label_source]].append(entry)

    cohort = []
    for label in sorted(per_class.keys()):
        sorted_entries = sorted(
            per_class[label],
            key=lambda e: e["n_vertices"],
            reverse=(mode == "graded"),
        )
        kept = 0
        for entry in sorted_entries:
            # Graded subdivision only adds vertices, so skip an entry already at or
            # above the target to avoid deduplication
            if mode == "graded" and entry["n_vertices"] >= fraction:
                continue
            sub = _subdivide_entry(entry, mode, fraction=fraction, rng=rng)
            if sub["n_vertices"] >= min_vertices:
                cohort.append(sub)
                kept += 1
                if kept == n_smallest:
                    break
        if kept < n_smallest:
            print(
                f"[WARN] Class {label!r}: only {kept} valid samples found "
                f"(requested {n_smallest}); pool exhausted."
            )

    return cohort


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate subdivided triangulation datasets."
    )
    parser.add_argument("--input", required=True, help="Raw MANTRA JSON file.")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for output JSON files."
    )
    parser.add_argument(
        "--prefix", required=True, help="Output filename prefix."
    )
    parser.add_argument(
        "--mode",
        default="full_barycentric",
        choices=sorted(_SUBDIVISION_FUNCTIONS),
    )
    parser.add_argument(
        "--n-levels",
        type=float,
        default=1.0,
        help="Subdivision levels; a fraction in (0, 1) for partial stellar "
        "subdivision; the target vertex count for mode=graded.",
    )
    parser.add_argument("--min-vertices", type=int, default=0)
    parser.add_argument("--label-source", default="name")
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=0,
        help="Pre-subdivision: drop classes with <= this many entries.",
    )
    parser.add_argument("--n-smallest", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    with open(args.input) as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} raw triangulations from {args.input}")

    outputs = generate_levels(
        data,
        mode=args.mode,
        n_levels=args.n_levels,
        min_vertices=args.min_vertices,
        label_source=args.label_source,
        min_class_count=args.min_class_count,
        n_smallest=args.n_smallest,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for key, entries in outputs.items():
        out_path = os.path.join(args.output_dir, f"{args.prefix}_{key}.json")
        with open(out_path, "w") as f:
            json.dump(entries, f)
        ns = [e["n_vertices"] for e in entries]
        print(f"[INFO] {key}: kept {len(entries)} -> {out_path}")
        if ns:
            print(
                f"       n_vertices: min={min(ns)}, max={max(ns)}, "
                f"mean={sum(ns) / len(ns):.1f}"
            )
        print_statistics(entries, args.label_source)


if __name__ == "__main__":
    main()
