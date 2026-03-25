"""Generate balanced dataset variants.

Usage::

    python scripts/generate_balanced.py 2_manifolds.json \
        2_manifolds_balanced.json --target-count 1000
    python scripts/generate_balanced.py 3_manifolds.json \
        3_manifolds_balanced.json --target-count 500 --dimension 3
"""

import argparse
import json

from mantra.augmentations.balancing import (
    balance_dataset,
    print_statistics,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate balanced dataset variants."
    )
    parser.add_argument("input", help="Path to input JSON dataset.")
    parser.add_argument("output", help="Path to output balanced JSON dataset.")
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        choices=[2, 3],
        help="Dimension of the manifolds (default: 2).",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=1000,
        help="Target count per class (default: 1000).",
    )
    parser.add_argument(
        "--n-moves",
        type=int,
        default=5,
        help="Number of Pachner moves per augmented sample " "(default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--no-topology-changes",
        action="store_true",
        help="Disable topology-changing operations for 2D.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip post-augmentation deduplication.",
    )
    parser.add_argument(
        "--dedup-max-rounds",
        type=int,
        default=10,
        help="Max dedup-regenerate rounds (default: 10).",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} entries from {args.input}")
    print("\nInput statistics:")
    print_statistics(dataset)

    print(f"\nBalancing to {args.target_count} per class...")
    balanced = balance_dataset(
        dataset,
        dimension=args.dimension,
        target_count=args.target_count,
        n_moves=args.n_moves,
        seed=args.seed,
        use_topology_changes=not args.no_topology_changes,
        dedup_max_rounds=0 if args.no_dedup else args.dedup_max_rounds,
        verbose=True,
    )

    print("\nOutput statistics:")
    print_statistics(balanced)

    with open(args.output, "w") as f:
        json.dump(balanced, f)

    print(f"\nWritten {len(balanced)} entries to {args.output}")


if __name__ == "__main__":
    main()
