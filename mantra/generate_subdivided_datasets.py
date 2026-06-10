"""Generate subdivided MANTRA dataset variants.

Applies repeated subdivision to every entry in a MANTRA-style raw triangulation
JSON file and writes one output JSON per subdivision level. Entries whose
subdivided vertex count is below ``--min-vertices`` are dropped from the output
(but kept in the running list so later levels are computed from the full
intermediate).

Three subdivision modes are supported:

* ``full_barycentric`` -- the classical barycentric subdivision (each face
  becomes a vertex; integer levels only).
* ``stellar`` -- stellar subdivision; supports a *fractional* level in (0, 1)
  to subdivide only a random fraction of the top-dimensional simplices.
* ``graded`` -- repeated stellar subdivision until a target vertex count is
  reached (``--n-levels`` is then read as that target count).

Optionally applies a pre-subdivision class-count filter (``--min-class-count``)
and/or selects the ``--n-smallest`` valid samples per class, mirroring the
filtering that the benchmarks training pipeline applies downstream but pushing
it upstream so the class set is deterministic across processes.

Output naming
-------------
For ``full_barycentric``/``stellar`` at *integer* levels the output filename
matches exactly what :class:`mantra.datasets.ManifoldTriangulations` expects for
``subdivision_level=L`` -- i.e. ``{dim}_manifolds{_balanced}_bary{L}.json`` (the
``_suffix`` convention from :mod:`mantra.datasets.base`). These files are
release-ready: upload them as ``.json.gz`` next to the base release and the
dataset class will auto-download them. Fractional and ``graded`` outputs use a
descriptive ``{prefix}`` name and are intended to be consumed via the dataset's
``local_path`` argument.

Example
-------
::

    python -m mantra.generate_subdivided_datasets \\
        --input 2_manifolds.json \\
        --output-dir ./subdivided \\
        --dimension 2 \\
        --n-levels 2 \\
        --mode full_barycentric \\
        --min-vertices 16 \\
        --min-class-count 0 \\
        --n-smallest 0
"""

import argparse
import os
import random
from collections import Counter, defaultdict

from mantra.datasets.base import _suffix
from mantra.subdivision import (
    barycentric_stellar_graded,
    barycentric_subdivision_raw,
    stellar_subdivision_raw,
)
from mantra.utils import store_triangulations


_SUBDIVISION_FUNCTIONS = {
    "stellar": stellar_subdivision_raw,
    "full_barycentric": barycentric_subdivision_raw,
    "graded": barycentric_stellar_graded,
}


def _subdivide_entry(entry, mode, fraction=1.0, rng=None):
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
    return [_subdivide_entry(e, mode, fraction=fraction, rng=rng) for e in entries]


def _level_tag(level):
    """Format a level for descriptive filenames: ints as '1', fractions as '0.5'."""
    return f"{level:g}"


def _output_path(output_dir, prefix, dimension, balanced, mode, level, fractional):
    """Build the output path for a given level.

    Integer ``full_barycentric``/``stellar`` levels use the canonical
    ``{dim}_manifolds{_balanced}_bary{L}.json`` name (shared with
    ``mantra.datasets.base._suffix``) so the result is release-ready and
    auto-downloadable via ``subdivision_level``. Fractional and graded outputs
    use the descriptive ``{prefix}`` name and are consumed via ``local_path``.
    """
    if mode == "graded":
        return os.path.join(output_dir, f"{prefix}_graded_n{_level_tag(level)}.json")
    if fractional:
        return os.path.join(output_dir, f"{prefix}_bary_{_level_tag(level)}.json")
    # Integer level, full_barycentric or stellar -> canonical release name.
    return os.path.join(
        output_dir, f"{dimension}_manifolds{_suffix(balanced, int(level))}.json"
    )


def _write(entries, path):
    """Write entries to ``path`` in the library's pretty JSON format.

    ``store_triangulations`` asserts at least one ``triangulation`` field, so an
    empty cohort is written as a literal empty list instead.
    """
    if not entries:
        with open(path, "w") as f:
            f.write("[]")
        return
    store_triangulations(entries, output=path)


def print_statistics(dataset, label_source="name"):
    """Print per-class n_vertices statistics for a list of entries."""
    class_entries = defaultdict(list)
    for entry in dataset:
        class_entries[entry[label_source]].append(entry)

    print(f"{'Class':<30} {'Count':>8} {'Min V':>8} {'Mean V':>8} {'Max V':>8}")
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


def filter_by_class_count(entries, label_source, min_count):
    """Drop entries whose ``label_source`` value occurs <= ``min_count`` times."""
    if min_count <= 0:
        return entries, Counter()
    counts = Counter(e[label_source] for e in entries)
    kept_labels = {lbl for lbl, c in counts.items() if c > min_count}
    filtered = [e for e in entries if e[label_source] in kept_labels]
    return filtered, counts


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate subdivided MANTRA dataset variants."
    )
    parser.add_argument("--input", required=True, help="Path to input JSON dataset.")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for the output JSON files."
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        choices=[2, 3],
        help="Manifold dimension; used to build canonical output names (default: 2).",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Mark outputs as the balanced variant in their canonical filename.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Descriptive prefix for fractional/graded outputs "
        "(default: '{dim}_manifolds').",
    )
    parser.add_argument(
        "--n-levels",
        type=float,
        default=1.0,
        help="Number of subdivision levels; a fraction in (0, 1) for partial "
        "stellar subdivision; the target vertex count for mode=graded.",
    )
    parser.add_argument(
        "--min-vertices",
        type=int,
        default=16,
        help="Drop subdivided entries below this vertex count (default: 16).",
    )
    parser.add_argument(
        "--label-source",
        default="name",
        help="Entry key used for class statistics/filtering (default: 'name').",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=100,
        help="Pre-subdivision: drop classes with <= this many entries "
        "(0 disables; default: 100).",
    )
    parser.add_argument(
        "--n-smallest",
        type=int,
        default=100,
        help="Keep only the N smallest valid samples per class "
        "(0 disables cohort selection; default: 100).",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(_SUBDIVISION_FUNCTIONS),
        default="stellar",
        help="Subdivision mode (default: stellar).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args(argv)


def main(argv=None):
    import json

    args = _parse_args(argv)
    prefix = args.prefix or f"{args.dimension}_manifolds"
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} raw triangulations from {args.input}")

    print(f"[INFO] Subdivision mode: {args.mode}")

    nl = float(args.n_levels)
    if nl <= 0 and args.mode != "graded":
        raise ValueError(f"n_levels must be > 0, got {nl}")
    fractional = nl < 1.0
    if fractional and args.mode != "stellar":
        raise ValueError(
            "Fractional n_levels (partial subdivision) only supported in "
            "mode='stellar'; full_barycentric breaks the SC condition under "
            "partial application."
        )
    n_full_levels = 0 if fractional or args.mode == "graded" else int(round(nl))
    if not fractional and abs(nl - n_full_levels) > 1e-9 and args.mode != "graded":
        raise ValueError(
            f"n_levels >= 1 must be integer-valued (got {nl}); fractional values "
            f"are only allowed in (0, 1)."
        )
    fraction = nl if fractional or args.mode == "graded" else 1.0
    print(
        f"[INFO] n_levels={nl} (fractional={fractional}, fraction={fraction}, "
        f"full_levels={n_full_levels}, seed={args.seed})"
    )

    if args.min_class_count > 0:
        before = len(data)
        data, _ = filter_by_class_count(data, args.label_source, args.min_class_count)
        kept_classes = sorted({e[args.label_source] for e in data})
        print(
            f"[INFO] Class filter ({args.label_source} > {args.min_class_count}): "
            f"{before} -> {len(data)} entries across {len(kept_classes)} classes"
        )
        print(f"       classes: {kept_classes}")

    rng = random.Random(args.seed)

    if args.n_smallest > 0:
        per_class = defaultdict(list)
        for entry in data:
            per_class[entry[args.label_source]].append(entry)

        cohort = []
        n_dropped = 0
        for label in sorted(per_class.keys()):
            sorted_entries = sorted(
                per_class[label],
                key=lambda e: e["n_vertices"],
                reverse=(args.mode == "graded"),  # descending if graded
            )
            kept = 0
            for entry in sorted_entries:
                sub = _subdivide_entry(entry, args.mode, fraction=fraction, rng=rng)
                if sub["n_vertices"] >= args.min_vertices:
                    cohort.append(sub)
                    kept += 1
                    if kept == args.n_smallest:
                        break
                else:
                    n_dropped += 1
            if kept < args.n_smallest:
                print(
                    f"[WARN] Class {label!r}: only {kept} valid samples found "
                    f"(requested {args.n_smallest}); pool exhausted."
                )

        first_level = fraction if (fractional or args.mode == "graded") else 1
        out_path = _output_path(
            args.output_dir, prefix, args.dimension, args.balanced,
            args.mode, first_level, fractional,
        )
        _write(cohort, out_path)
        ns = [e["n_vertices"] for e in cohort]
        print(
            f"[INFO] Level {_level_tag(first_level)}: kept {len(cohort)} smallest valid "
            f"({args.n_smallest}/class, skipped {n_dropped} below "
            f"{args.min_vertices} vertices) -> {out_path}"
        )
        if ns:
            print(
                f"       n_vertices: min={min(ns)}, max={max(ns)}, "
                f"mean={sum(ns) / len(ns):.1f}"
            )
        print_statistics(cohort, args.label_source)

        if fractional or args.mode == "graded":
            return

        current = cohort
        for level in range(2, n_full_levels + 1):
            current = subdivide_once(current, args.mode)
            below = [e for e in current if e["n_vertices"] < args.min_vertices]
            if below:
                print(
                    f"[WARN] Level {level}: {len(below)} entries below "
                    f"{args.min_vertices} vertices after subdivision (unexpected)."
                )
            out_path = _output_path(
                args.output_dir, prefix, args.dimension, args.balanced,
                args.mode, level, fractional,
            )
            _write(current, out_path)
            ns = [e["n_vertices"] for e in current]
            print(f"[INFO] Level {level}: kept {len(current)} -> {out_path}")
            if ns:
                print(
                    f"       n_vertices: min={min(ns)}, max={max(ns)}, "
                    f"mean={sum(ns) / len(ns):.1f}"
                )
            print_statistics(current, args.label_source)
        return

    if fractional:
        current = subdivide_once(data, args.mode, fraction=fraction, rng=rng)
        filtered = [e for e in current if e["n_vertices"] >= args.min_vertices]
        n_dropped = len(current) - len(filtered)
        out_path = _output_path(
            args.output_dir, prefix, args.dimension, args.balanced,
            args.mode, fraction, fractional,
        )
        _write(filtered, out_path)
        ns = [e["n_vertices"] for e in filtered]
        print(
            f"[INFO] Level {_level_tag(fraction)}: kept {len(filtered)} "
            f"(dropped {n_dropped} below {args.min_vertices} vertices) -> {out_path}"
        )
        if ns:
            print(
                f"       n_vertices: min={min(ns)}, max={max(ns)}, "
                f"mean={sum(ns) / len(ns):.1f}"
            )
        print_statistics(filtered, args.label_source)
        return

    current = data
    for level in range(1, n_full_levels + 1):
        current = subdivide_once(current, args.mode)
        filtered = [e for e in current if e["n_vertices"] >= args.min_vertices]
        n_dropped = len(current) - len(filtered)
        out_path = _output_path(
            args.output_dir, prefix, args.dimension, args.balanced,
            args.mode, level, fractional,
        )
        _write(filtered, out_path)
        ns = [e["n_vertices"] for e in filtered]
        print(
            f"[INFO] Level {level}: kept {len(filtered)} "
            f"(dropped {n_dropped} below {args.min_vertices} vertices) -> {out_path}"
        )
        if ns:
            print(
                f"       n_vertices: min={min(ns)}, max={max(ns)}, "
                f"mean={sum(ns) / len(ns):.1f}"
            )
        print_statistics(filtered, args.label_source)


if __name__ == "__main__":
    main()
