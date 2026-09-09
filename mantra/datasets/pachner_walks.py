"""Dataset of Pachner walks starting from MANTRA triangulations."""

import random
from typing import List

from torch_geometric.data import Data
from tqdm import tqdm

from mantra.datasets.mantra_dataset import MantraDataset
from mantra.utils import Triangulation

# Distinct seeds per split keep the walks of the splits independent of
# each other and of the order in which the splits are processed.
WALK_SEED_OFFSETS = {"train": 1, "val": 2, "test": 3}


class PachnerWalkDataset(MantraDataset):
    """MANTRA splits in which every entry is expanded into a Pachner walk.

    The train, val and test splits contain the same base entries as
    :class:`MantraDataset` with the same ``seed``, but every base
    triangulation ``T_0`` is followed by the snapshots ``T_1, ...,
    T_K`` of a random Pachner walk started from it, each snapshot a
    separate ``Data`` object. Entries are ordered base-major, i.e. all
    snapshots of the first base entry precede those of the second. The
    ``ood`` split is the same as for :class:`MantraDataset`.

    With ``walk_length=0`` the dataset is identical to
    :class:`MantraDataset`, including its cached files. Otherwise every
    entry of the expanded splits carries two additional attributes:

    ``walk_base``
        Position of its base entry within the split, starting at 0. The
        name deliberately avoids the substring ``index``: PyTorch
        Geometric increments attributes whose key contains it when
        collating a batch, which would renumber the walks.
    ``walk_step``
        Index ``k`` of the snapshot ``T_k`` on the walk; the base entry
        itself has ``walk_step = 0`` and keeps its ``id``, later
        snapshots get the id ``"<base id>_walk_<k>"``.

    ``triangulation`` and ``n_vertices`` describe the snapshot; all
    other attributes (``name``, ``betti_numbers``, ``orientable``,
    ``genus``, ...) are copied from the base entry, since Pachner moves
    preserve the homeomorphism type.

    Two snapshots ``a`` and ``b`` on the same walk are separated by
    ``|walk_step_a - walk_step_b| * moves_per_step`` random moves.
    This is an upper bound on their Pachner distance (the minimal
    number of moves connecting them), not the distance itself: a walk
    can undo its own moves. Treat it as a monotone proxy for the
    distance. Snapshots of different walks are not related by any
    known number of moves.

    Parameters
    ----------
    walk_length : int
        Number ``K`` of snapshots taken after the base triangulation.
        0 disables the expansion.
    moves_per_step : int
        Number of random Pachner moves between consecutive snapshots.
    walk_seed : int or None
        Seed of the random walks. Defaults to ``seed``, so changing
        the split seed also changes the walks; set it explicitly to
        draw different walks over the same splits.
    seed : int
        Split seed, see :class:`MantraDataset`.
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
        walk_seed: int | None = None,
        seed: int = 42,
        **kwargs,
    ):
        if walk_length < 0:
            raise ValueError(f"walk_length must be >= 0, got {walk_length}")
        if moves_per_step < 1:
            raise ValueError(
                f"moves_per_step must be >= 1, got {moves_per_step}"
            )
        self.walk_length = walk_length
        self.moves_per_step = moves_per_step
        self.walk_seed = seed if walk_seed is None else walk_seed
        super().__init__(root, split_type, seed=seed, **kwargs)

    def _walk_file_suffix(self):
        """Suffix encoding the walk parameters; empty without walks."""
        if self.walk_length == 0:
            return ""
        return (
            f"_walk{self.walk_length}x{self.moves_per_step}"
            f"_ws{self.walk_seed}"
        )

    @property
    def processed_file_names(self):
        """Return processed file names.

        The walk parameters are appended to the train, val and test
        file names only; the OOD file is the same as for
        :class:`MantraDataset` and can be shared with it.
        """
        names = super().processed_file_names
        suffix = self._walk_file_suffix()
        return [
            f"{name.removesuffix('.pt')}{suffix}.pt" for name in names[:3]
        ] + names[3:]

    def _expand_split(self, split_type: str, data_list: List[Data]):
        """Expand every entry of the split into its Pachner walk."""
        if self.walk_length == 0:
            return data_list

        rng = random.Random(self.walk_seed + WALK_SEED_OFFSETS[split_type])
        expanded = []
        for walk_base, data in enumerate(
            tqdm(data_list, desc=f"Pachner walks ({split_type})")
        ):
            triangulation = Triangulation.from_list(
                data.triangulation, rng=rng
            )
            snapshots = triangulation.random_walk(
                self.walk_length, self.moves_per_step
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
