"""Dataset of Pachner walks starting from MANTRA triangulations."""

import random
from typing import List, Sequence

from torch_geometric.data import Data
from tqdm import tqdm

from mantra.datasets.mantra_dataset import (
    DEFAULT_PACHNER_WEIGHTS,
    MantraDataset,
    pachner_walk_to,
)
from mantra.utils import Triangulation

# Distinct seeds per split keep the walks of the splits independent of
# each other and of the order in which the splits are processed.
WALK_SEED_OFFSETS = {"train": 1, "val": 2, "test": 3}
# Splits the size-targeted walks expand by default: the test split stays
# the plain release split, so `test/*` metrics remain comparable across
# runs, and the OOD splits cover the sizes.
DEFAULT_SIZE_SPLITS = ("train", "val")


class PachnerWalkDataset(MantraDataset):
    """MANTRA splits in which every entry is expanded into a Pachner walk.

    Two expansions exist, mutually exclusive.

    **Step walks** (``walk_length > 0``). Every base triangulation
    ``T_0`` of the train, val and test splits of :class:`MantraDataset`
    is followed by the snapshots ``T_1, ..., T_K`` of a random Pachner
    walk started from it, each a separate ``Data`` object with
    ``triangulation`` and ``n_vertices`` of the snapshot and every other
    attribute copied from the base entry.

    **Size walks** (``size_targets``). Every base triangulation of the
    splits in ``size_splits`` is followed by one refining walk per
    target vertex count ``n`` in ``size_targets``, stopped exactly at
    ``n`` vertices (as the ``target`` argument of the ``pachner`` OOD
    split does) and optionally mixed by ``size_mix`` 2-2 flips per
    vertex. With ``keep_base=False`` the base entry itself is dropped,
    so the split contains only the target sizes. Every class appears at
    every target size with the same multiplicity, so the target sizes
    carry no label information (the base sizes do: the smallest
    triangulations of the release belong to few classes).

    Two attributes identify the walk in both modes:

    ``walk_base``
        Position of the base entry within the split. The name avoids
        the substring ``index``, which PyTorch Geometric increments
        when collating a batch.
    ``walk_step``
        Index ``k`` of the snapshot: ``0`` is the base entry, ``k > 0``
        gets the id ``"<base id>_walk_<k>"`` (step walks) or
        ``"<base id>_size_<n>"`` (size walks, ``k`` the position of
        ``n`` in ``size_targets`` plus one).

    For step walks ``|walk_step_a - walk_step_b| * moves_per_step``
    random moves separate two snapshots of one walk. This is an upper
    bound on their Pachner distance, not the distance itself, since a
    walk can undo its own moves. The ``ood`` split and, without either
    expansion, all splits are identical to :class:`MantraDataset`,
    cache files included.

    Parameters
    ----------
    walk_length : int
        Number ``K`` of snapshots taken after the base triangulation.
        0 disables the step walks.
    moves_per_step : int
        Number of random Pachner moves between consecutive snapshots.
    move_weights : sequence of float or None
        Relative weights of the ``(flip, subdivide, coarsen)`` moves,
        i.e. 2-2, 1-3 and 3-1 in 2D; ``None`` means equal weights for
        step walks and ``(1, 1, 0)`` (refining) for size walks, which
        need a positive 1-3 weight and a zero 3-1 weight to reach their
        target.
    size_targets : sequence of int or None
        Vertex counts of the size walks, strictly increasing; ``None``
        disables them. Every target must exceed ``max_vertices`` (if
        set) and the vertex count of every source, as the OOD
        ``target`` must.
    size_splits : sequence of str
        Splits the size walks expand (default train and val).
    keep_base : bool
        Keep the base entry (step 0) next to its size walks.
    size_mix : float
        Extra 2-2 flips per vertex after a size walk reaches its target.
    seed : int
        Split seed, see :class:`MantraDataset`; also seeds the walks.
    kwargs : dict
        All other arguments of :class:`MantraDataset`.
    """

    def __init__(
        self,
        root,
        split_type: str,
        *,
        walk_length: int = 0,
        moves_per_step: int = 1,
        move_weights: Sequence[float] | None = None,
        size_targets: Sequence[int] | None = None,
        size_splits: Sequence[str] = DEFAULT_SIZE_SPLITS,
        keep_base: bool = True,
        size_mix: float = 0.0,
        seed: int = 42,
        **kwargs,
    ):
        if walk_length < 0:
            raise ValueError(f"walk_length must be >= 0, got {walk_length}")
        if moves_per_step < 1:
            raise ValueError(
                f"moves_per_step must be >= 1, got {moves_per_step}"
            )
        if move_weights is not None:
            move_weights = tuple(float(w) for w in move_weights)
            if len(move_weights) != 3:
                raise ValueError(
                    "move_weights needs three entries (flip, subdivide, "
                    f"coarsen), got {list(move_weights)}"
                )
        if size_targets is not None:
            size_targets = self._validate_size_targets(
                size_targets,
                kwargs.get("max_vertices"),
                kwargs.get("dimension", 2),
            )
            if walk_length > 0:
                raise ValueError(
                    "size_targets and walk_length > 0 are two expansions of "
                    "the same split; set one of them"
                )
            weights = (
                move_weights
                if move_weights is not None
                else DEFAULT_PACHNER_WEIGHTS
            )
            if weights[1] <= 0 or weights[2] > 0:
                raise ValueError(
                    "Size walks need a positive 1-3 weight and a zero 3-1 "
                    f"weight to reach their target, got {list(weights)}"
                )
            unknown = set(size_splits) - set(WALK_SEED_OFFSETS)
            if unknown or not size_splits:
                raise ValueError(
                    f"size_splits must be a non-empty subset of "
                    f"{sorted(WALK_SEED_OFFSETS)}, got {list(size_splits)}"
                )
            if size_mix < 0:
                raise ValueError(f"size_mix must be >= 0, got {size_mix}")
        self.walk_length = walk_length
        self.moves_per_step = moves_per_step
        self.move_weights = move_weights
        self.size_targets = size_targets
        self.size_splits = tuple(size_splits)
        self.keep_base = bool(keep_base)
        self.size_mix = float(size_mix)
        super().__init__(root, split_type, seed=seed, **kwargs)

    @staticmethod
    def _validate_size_targets(size_targets, max_vertices, dimension):
        if dimension != 2:
            raise NotImplementedError("Size walks are 2D only")
        targets = []
        for target in size_targets:
            if isinstance(target, bool) or not isinstance(target, int):
                raise ValueError(f"size_targets must be ints, got {target!r}")
            targets.append(int(target))
        if len(targets) == 0:
            raise ValueError("size_targets must not be empty")
        if targets != sorted(set(targets)):
            raise ValueError(
                f"size_targets must be strictly increasing, got {targets}"
            )
        if max_vertices is not None and targets[0] <= max_vertices:
            raise ValueError(
                f"size_targets ({targets}) must all be strictly greater "
                f"than max_vertices ({max_vertices})"
            )
        return tuple(targets)

    @property
    def size_walk_weights(self):
        """``(flip, subdivide, coarsen)`` weights of the size walks."""
        if self.move_weights is not None:
            return self.move_weights
        return DEFAULT_PACHNER_WEIGHTS

    def _walk_file_suffix(self, split_type: str):
        """Suffix encoding the walk parameters; empty without walks."""
        if self.walk_length > 0:
            suffix = (
                f"_walk{self.walk_length}x{self.moves_per_step}_ws{self.seed}"
            )
            if self.move_weights is not None:
                weights = "-".join(f"{w:g}" for w in self.move_weights)
                suffix += f"_mw{weights}"
            return suffix
        if self.size_targets is not None and split_type in self.size_splits:
            suffix = "_size" + "-".join(str(n) for n in self.size_targets)
            if not self.keep_base:
                suffix += "_nb"
            if self.size_mix:
                suffix += f"_mix{self.size_mix:g}"
            if self.size_walk_weights != DEFAULT_PACHNER_WEIGHTS:
                weights = "-".join(f"{w:g}" for w in self.size_walk_weights)
                suffix += f"_mw{weights}"
            return suffix + f"_ws{self.seed}"
        return ""

    @property
    def processed_file_names(self):
        """Walk parameters go into the train, val and test names only."""
        names = super().processed_file_names
        return [
            f"{name.removesuffix('.pt')}{self._walk_file_suffix(split)}.pt"
            for name, split in zip(names[:3], WALK_SEED_OFFSETS)
        ] + names[3:]

    def _expand_split(self, split_type: str, data_list: List[Data]):
        """Expand every entry of the split into its Pachner walk."""
        if self.walk_length > 0:
            return self._expand_step_walks(split_type, data_list)
        if self.size_targets is not None and split_type in self.size_splits:
            return self._expand_size_walks(split_type, data_list)
        return data_list

    def _expand_step_walks(self, split_type: str, data_list: List[Data]):
        rng = random.Random(self.seed + WALK_SEED_OFFSETS[split_type])
        expanded = []
        for walk_base, data in enumerate(
            tqdm(data_list, desc=f"Pachner walks ({split_type})")
        ):
            triangulation = Triangulation.from_list(
                data.triangulation, rng=rng
            )
            snapshots = triangulation.random_walk(
                self.walk_length, self.moves_per_step, self.move_weights
            )
            for step, simplices in enumerate(snapshots):
                entry = Data(**data.to_dict())
                entry.walk_base = walk_base
                entry.walk_step = step
                if step > 0:
                    entry.id = f"{data.id}_walk_{step}"
                    entry.triangulation = simplices
                    entry.n_vertices = len({v for s in simplices for v in s})
                expanded.append(entry)
        return expanded

    def _expand_size_walks(self, split_type: str, data_list: List[Data]):
        """One refining walk per (entry, target), each from the base."""
        rng = random.Random(self.seed + WALK_SEED_OFFSETS[split_type])
        weights = self.size_walk_weights
        expanded = []
        for walk_base, data in enumerate(
            tqdm(data_list, desc=f"Size walks ({split_type})")
        ):
            if self.keep_base:
                entry = Data(**data.to_dict())
                entry.walk_base = walk_base
                entry.walk_step = 0
                expanded.append(entry)
            for step, target in enumerate(self.size_targets, 1):
                triangle = Triangulation.from_list(data.triangulation, rng=rng)
                pachner_walk_to(triangle, target, weights, self.size_mix)
                entry = Data(**data.to_dict())
                entry.walk_base = walk_base
                entry.walk_step = step
                entry.id = f"{data.id}_size_{target}"
                entry.triangulation = triangle.to_list()
                entry.n_vertices = triangle.n_vertices
                expanded.append(entry)
        return expanded
