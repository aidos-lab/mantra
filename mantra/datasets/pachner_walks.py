"""Dataset of Pachner walks starting from MANTRA triangulations."""

import random
from typing import List, Sequence

from torch_geometric.data import Data
from tqdm import tqdm

from mantra.datasets.mantra_dataset import MantraDataset
from mantra.utils import Triangulation

# Distinct seeds per split keep the walks of the splits independent of
# each other and of the order in which the splits are processed.
WALK_SEED_OFFSETS = {"train": 1, "val": 2, "test": 3}


class PachnerWalkDataset(MantraDataset):
    """MANTRA splits in which every entry is expanded into a Pachner walk.

    Every base triangulation ``T_0`` of the train, val and test splits
    of :class:`MantraDataset` is followed by the snapshots ``T_1, ...,
    T_K`` of a random Pachner walk started from it, each a separate
    ``Data`` object with ``triangulation`` and ``n_vertices`` of the
    snapshot and every other attribute copied from the base entry. Two
    attributes identify the walk:

    ``walk_base``
        Position of the base entry within the split. The name avoids
        the substring ``index``, which PyTorch Geometric increments
        when collating a batch.
    ``walk_step``
        Index ``k`` of the snapshot; ``k > 0`` gets the id
        ``"<base id>_walk_<k>"``.

    ``|walk_step_a - walk_step_b| * moves_per_step`` random moves
    separate two snapshots of one walk. This is an upper bound on their
    Pachner distance, not the distance itself, since a walk can undo
    its own moves. The ``ood`` split and, with ``walk_length=0``, all
    splits are identical to :class:`MantraDataset`, cache files
    included.

    Parameters
    ----------
    walk_length : int
        Number ``K`` of snapshots taken after the base triangulation.
        0 disables the expansion.
    moves_per_step : int
        Number of random Pachner moves between consecutive snapshots.
    move_weights : sequence of float or None
        Relative weights of the ``(flip, subdivide, coarsen)`` moves,
        i.e. 2-2, 1-3 and 3-1 in 2D; ``None`` means equal weights and
        ``(1, 1, 0)`` gives refining walks.
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
        self.walk_length = walk_length
        self.moves_per_step = moves_per_step
        self.move_weights = move_weights
        super().__init__(root, split_type, seed=seed, **kwargs)

    def _walk_file_suffix(self):
        """Suffix encoding the walk parameters; empty without walks."""
        if self.walk_length == 0:
            return ""
        suffix = f"_walk{self.walk_length}x{self.moves_per_step}_ws{self.seed}"
        if self.move_weights is not None:
            weights = "-".join(f"{w:g}" for w in self.move_weights)
            suffix += f"_mw{weights}"
        return suffix

    @property
    def processed_file_names(self):
        """Walk parameters go into the train, val and test names only."""
        names = super().processed_file_names
        suffix = self._walk_file_suffix()
        return [
            f"{name.removesuffix('.pt')}{suffix}.pt" for name in names[:3]
        ] + names[3:]

    def _expand_split(self, split_type: str, data_list: List[Data]):
        """Expand every entry of the split into its Pachner walk."""
        if self.walk_length == 0:
            return data_list

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
