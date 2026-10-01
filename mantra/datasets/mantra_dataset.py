import math
import random
import warnings
from collections import defaultdict
from enum import Enum
from typing import List

import numpy as np
from torch_geometric.data import (
    Data,
)
from tqdm import tqdm

from mantra.datasets.mantra import ManifoldTriangulations
from mantra.datasets.utils import filter_by_class_count, make_split_index
from mantra.utils import Triangulation
from mantra.utils.balancing import balance_dataset

SPLIT_TYPES = ["train", "val", "test", "ood"]
DEFAULT_SPLIT_PROPORTIONS = [0.6, 0.2, 0.2]


class SubdivisionType(Enum):
    STELLAR = 1
    GRADED = 2
    BARYCENTRIC = 3
    NONE = 4
    PACHNER = 5

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(sub_name: str):
        for sub in SubdivisionType:
            if str(sub).lower() == sub_name.lower():
                return sub
        raise ValueError(f"There is no Subdivision with name {sub_name}")


# Arguments of the OOD subdivisions, see MantraDataset.
SUBDIVISION_KEYS = {
    SubdivisionType.BARYCENTRIC: {"round"},
    SubdivisionType.STELLAR: {"fraction"},
    SubdivisionType.GRADED: {"graded_vertex_number"},
    SubdivisionType.PACHNER: {"target", "match", "move_weights", "mix"},
}
# Pachner walks of the OOD split (2D only): (flip, subdivide, coarsen)
# weights of the size-changing phase, and the number of moves allowed
# per vertex the walk has to gain before it is declared stuck.
DEFAULT_PACHNER_WEIGHTS = (1.0, 1.0, 0.0)
PACHNER_MOVE_BUDGET = 100


def pachner_walk_to(triangle, target, weights, mix=0):
    """Walk ``triangle`` in place to ``target`` vertices, then mix it.

    Random Pachner moves drawn with ``weights`` (``(flip, subdivide,
    coarsen)``) until the vertex count equals ``target`` (``None``: no
    size-changing phase), then ``mix`` 2-2 flips per vertex, which keep
    the size. A triangulation without a flippable edge, such as the
    tetrahedral sphere, ends the mixing early. Shared by the OOD walks
    of :class:`MantraDataset` and the size-targeted training walks of
    :class:`~mantra.datasets.PachnerWalkDataset`.

    Raises
    ------
    ValueError
        If ``target`` is below the vertex count of ``triangle``.
    RuntimeError
        If the walk does not reach ``target`` within
        ``PACHNER_MOVE_BUDGET`` moves per vertex it has to gain.
    """
    if target is not None:
        n = triangle.n_vertices
        if target < n:
            raise ValueError(
                f"Pachner target ({target}) is below the vertex count "
                f"({n}) of the source"
            )
        budget = PACHNER_MOVE_BUDGET * (target - n + 1)
        moves = 0
        while triangle.n_vertices != target:
            if moves >= budget or not triangle.random_pachner_move(weights):
                raise RuntimeError(
                    f"Pachner walk did not reach {target} vertices "
                    f"within {budget} moves"
                )
            moves += 1

    n_flips = round(mix * triangle.n_vertices)
    for _ in range(n_flips):
        if not triangle.flip_edge():
            break


class MantraDataset(ManifoldTriangulations):
    """Dataset of manifold triangulations from the MANTRA benchmark
    with subdivisions of the test set as an additional OOD split.
    """

    def __init__(
        self,
        root,
        split_type: str,
        dimension: int = 2,
        version: str = "latest",
        balanced: bool = False,
        name: str | None = None,
        local_path=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
        seed: int = 42,
        division_type: str = "none",
        min_sample_per_class: int | None = None,
        split_proportions: List[float] = DEFAULT_SPLIT_PROPORTIONS,
        stratified: bool = False,
        max_vertices: int | None = None,
        n_moves: int = 1,
        target_count: int = 10,
        use_surgery: bool = False,
        max_ood_size_per_class: int | None = None,
        relabel: bool = False,
        **kwargs,
    ):
        """
        Create a new dataset of manifold triangulations.

        Parameters
        ----------
        split_type: str
            Type of the split in [train, val, test, ood].
        division_type : str
            Type of division to apply to the triangulations. Options are
            barycentric, graded, stellar and pachner (a random Pachner
            walk instead of a subdivision).
        min_sample_per_class : int or None
            If the initial classes should be filtered before constructing the
            subdivisions.
        split_proportions : List[float]
            Proportional split in terms of [train, val, test]. Must sum
            to 1.
        stratified : bool
            If to use stratified splitting (by manifold class name).
        max_vertices : int or None
            If set, drop all triangulations with more than this many
            vertices before splitting, so train/val/test only contain
            triangulations with at most ``max_vertices`` vertices. In
            combination with a graded subdivision this guarantees that
            every OOD sample is strictly larger than any in-distribution
            sample, since ``graded_vertex_number`` must exceed ``max_vertices``.
        max_ood_size_per_class : int or None
            If set, oversample and trim the OOD split so that every
            class contains exactly this many samples (classes without
            eligible test-set sources are skipped). Oversampling draws
            additional randomized subdivisions from the same test-set
            sources; it only yields distinct triangulations for
            randomized subdivisions (graded, or stellar with
            ``fraction`` < 1), while barycentric subdivision is
            deterministic and produces exact duplicates.
        relabel : bool
            Rename the vertices of every OOD entry by a random
            permutation after the subdivision or walk. The entry then
            leaves the canonical labelling of the release without any
            further change; with ``division_type="pachner"`` and no other
            argument it is an isomorphic copy of its source.
        kwargs : Dict
            Arguments of the subdivision, checked against the division
            type.

            - barycentric: ``round``, number of rounds (default 1).
            - stellar: ``fraction`` of top simplices to subdivide
              (default 1.0).
            - graded: ``graded_vertex_number``, the vertex count every
              OOD entry is grown to (required, must exceed
              ``max_vertices``).
            - pachner (2D only), a random Pachner walk from the source:
              ``target``, the vertex count it stops at, or ``match``,
              a dict with the ``division_type`` and arguments of another
              subdivision, whose vertex count on the same source is the
              target (so the two OOD splits are matched in size entry
              by entry); neither keeps the size. ``move_weights``, the
              ``(flip, subdivide, coarsen)`` weights of that walk
              (default ``(1, 1, 0)``), and ``mix``, extra 2-2 flips per
              vertex applied afterwards (default 0), which change the
              local structure but not the size.
        """
        if split_type not in SPLIT_TYPES:
            raise ValueError(
                f"split_type must be one of {SPLIT_TYPES}, got '{split_type}'"
            )
        if len(split_proportions) != 3 or not math.isclose(
            sum(split_proportions), 1.0
        ):
            raise ValueError(
                "split_proportions must be [train, val, test] summing to 1, "
                f"got {split_proportions}"
            )

        self.stratified = stratified
        self.split_type = split_type
        self.split_proportions = split_proportions
        self.min_sample_per_class = min_sample_per_class
        self.division_type = SubdivisionType.from_str(division_type)
        self.max_ood_size_per_class = max_ood_size_per_class
        self.relabel = bool(relabel)
        self.kwargs = kwargs

        if balanced and min_sample_per_class:
            warnings.warn(
                "balanced=True equalizes the classes before "
                "min_sample_per_class is applied, so the filter can "
                "re-imbalance or drop classes again."
            )
        self._validate_subdivision(
            self.division_type, kwargs, max_vertices, dimension
        )

        super().__init__(
            root,
            version,
            dimension,
            name,
            balanced,
            local_path,
            transform,
            pre_transform,
            pre_filter,
            force_reload,
            seed,
            max_vertices,
            n_moves,
            target_count,
            use_surgery,
        )

    @classmethod
    def _validate_subdivision(cls, division_type, kwargs, max_vertices, dim):
        """Reject unknown, missing and unsafe subdivision arguments."""
        if division_type == SubdivisionType.NONE:
            allowed = set()
        else:
            allowed = SUBDIVISION_KEYS[division_type]
        if unknown := set(kwargs) - allowed:
            raise ValueError(
                f"'{division_type}' does not take {sorted(unknown)}; "
                f"its arguments are {sorted(allowed)}"
            )

        if division_type == SubdivisionType.GRADED:
            if "graded_vertex_number" not in kwargs:
                raise ValueError(
                    "Graded subdivision requires 'graded_vertex_number': "
                    "the number of vertices every OOD entry is grown to."
                )
            cls._check_above_max_vertices(
                "graded_vertex_number",
                kwargs["graded_vertex_number"],
                max_vertices,
            )

        if division_type == SubdivisionType.PACHNER:
            if dim != 2:
                raise NotImplementedError("Pachner OOD splits are 2D only")
            if "target" in kwargs and "match" in kwargs:
                raise ValueError(
                    "Pachner walks take 'target' or 'match', not both"
                )
            if "target" in kwargs:
                cls._check_above_max_vertices(
                    "target", kwargs["target"], max_vertices
                )
            if "match" in kwargs:
                match = dict(kwargs["match"])
                if "division_type" not in match:
                    raise ValueError(
                        "'match' needs the 'division_type' of the subdivision "
                        f"to match and its arguments, got {kwargs['match']!r}"
                    )
                matched = SubdivisionType.from_str(match.pop("division_type"))
                if matched in (SubdivisionType.NONE, SubdivisionType.PACHNER):
                    raise ValueError(f"Cannot match the size of '{matched}'")
                cls._validate_subdivision(matched, match, max_vertices, dim)
            weights = kwargs.get("move_weights", DEFAULT_PACHNER_WEIGHTS)
            if len(weights) != 3 or min(weights) < 0:
                raise ValueError(
                    "move_weights needs three non-negative entries "
                    f"(flip, subdivide, coarsen), got {list(weights)}"
                )
            if ("target" in kwargs or "match" in kwargs) and weights[1] <= 0:
                raise ValueError(
                    "A Pachner walk with a size target needs a positive "
                    "weight for the 1-3 move"
                )
            if kwargs.get("mix", 0) < 0:
                raise ValueError(f"mix must be >= 0, got {kwargs['mix']}")

    @staticmethod
    def _check_above_max_vertices(name, value, max_vertices):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an int, got {value!r}")
        if max_vertices is not None and value <= max_vertices:
            raise ValueError(
                f"{name} ({value}) must be strictly greater than "
                f"max_vertices ({max_vertices}); otherwise OOD entries are "
                "not guaranteed to be larger than the train/val/test ones."
            )

    def _load_index(self):
        """Load the processed file matching ``split_type``."""
        return SPLIT_TYPES.index(self.split_type)

    def _split_file_suffix(self):
        """Suffix encoding parameters that change the train/val/test data.

        The vertex cap needs no entry here: the parent class encodes
        ``max_vertices`` into the processed directory itself.
        """
        parts = []
        if self.min_sample_per_class:
            parts.append(f"ccf{self.min_sample_per_class}")
        if self.split_proportions != DEFAULT_SPLIT_PROPORTIONS:
            parts.append(
                "sp" + "-".join(str(p) for p in self.split_proportions)
            )
        if self.stratified:
            parts.append("strat")
        return "_" + "_".join(parts) if parts else ""

    @staticmethod
    def _subdivision_str(division_type, kwargs):
        """``<type>_<arguments>`` of a subdivision, as used in file names."""
        if division_type == SubdivisionType.BARYCENTRIC:
            args = f"{kwargs.get('round', 1)}"
        elif division_type == SubdivisionType.STELLAR:
            args = f"{kwargs.get('fraction', 1)}"
        elif division_type == SubdivisionType.GRADED:
            args = f"{kwargs['graded_vertex_number']}"
        else:
            args = MantraDataset._pachner_str(kwargs)
        return f"{division_type}_{args}"

    @staticmethod
    def _pachner_str(kwargs):
        """Arguments of a Pachner walk, as used in file names."""
        if "target" in kwargs:
            args = f"t{kwargs['target']}"
        elif "match" in kwargs:
            match = dict(kwargs["match"])
            matched = SubdivisionType.from_str(match.pop("division_type"))
            args = "m-" + MantraDataset._subdivision_str(matched, match)
        else:
            args = "n"
        if args != "n":  # the weights only act while the walk has a target
            weights = kwargs.get("move_weights", DEFAULT_PACHNER_WEIGHTS)
            args += "_w" + "-".join(f"{w:g}" for w in weights)
        if kwargs.get("mix", 0):
            args += f"_mix{kwargs['mix']:g}"
        return args

    def _build_ood_str(self):
        if self.division_type == SubdivisionType.NONE:
            return str(self.division_type)

        ood_str = self._subdivision_str(self.division_type, self.kwargs)
        if self.relabel:
            ood_str += "_rl"
        if self.max_ood_size_per_class is not None:
            ood_str += f"_cap{self.max_ood_size_per_class}"
        # "ss": all OOD sources are drawn before any subdivision (see
        # _build_ood_split). Caches without the marker predate that and
        # are not reused.
        return ood_str + "_ss"

    @property
    def processed_file_names(self):
        """Return process file names.

        Stores the processed data in a file. If this file is present in the
        `processed` folder, processing will typically be skipped.
        """
        suffix = self._split_file_suffix()
        base_files = []
        for split_type in SPLIT_TYPES[:3]:
            file_str = f"{split_type}{suffix}.pt"
            base_files.append(file_str)

        ood_file: str = f"ood_{self._build_ood_str()}{suffix}.pt"
        base_files.append(ood_file)

        return base_files

    @staticmethod
    def _apply_subdivision(triangle, division_type, kwargs):
        """Subdivide ``triangle`` in place as ``division_type`` prescribes."""
        if division_type == SubdivisionType.BARYCENTRIC:
            for _ in range(kwargs.get("round", 1)):
                triangle.barycentric_subdivision()
        elif division_type == SubdivisionType.STELLAR:
            triangle.stellar_subdivision(fraction=kwargs.get("fraction", 1.0))
        elif division_type == SubdivisionType.GRADED:
            triangle.graded_subdivision(
                over_vrtx_cnt=kwargs["graded_vertex_number"]
            )
        else:
            raise ValueError(f"'{division_type}' is not a subdivision")

    def _pachner_target(self, data):
        """Vertex count the Pachner walk of ``data`` stops at, or None."""
        if "target" in self.kwargs:
            return int(self.kwargs["target"])
        match = self.kwargs.get("match")
        if match is None:
            return None
        match = dict(match)
        matched = SubdivisionType.from_str(match.pop("division_type"))
        # The vertex count of every subdivision is a function of the
        # source alone (graded: the target; stellar: a rounded fraction
        # of the triangles; barycentric: all faces), so a throwaway
        # generator gives the size of the matched OOD entry exactly.
        probe = Triangulation.from_list(
            data.triangulation, rng=random.Random(0)
        )
        self._apply_subdivision(probe, matched, match)
        return probe.n_vertices

    def _pachner_walk(self, triangle, data):
        """Walk ``triangle`` (built from ``data``) as the kwargs prescribe.

        Random moves with ``move_weights`` until the vertex count equals
        the target (none: no size-changing phase), then ``mix`` 2-2
        flips per vertex, which keep the size. See :func:`pachner_walk_to`.
        """
        weights = tuple(
            self.kwargs.get("move_weights", DEFAULT_PACHNER_WEIGHTS)
        )
        pachner_walk_to(
            triangle,
            self._pachner_target(data),
            weights,
            self.kwargs.get("mix", 0),
        )

    def _subdivide_entry(self, data, rng, tag):
        """Return a copy of ``data`` with the subdivided triangulation."""
        triangle = Triangulation.from_list(data.triangulation, rng=rng)

        if self.division_type == SubdivisionType.PACHNER:
            self._pachner_walk(triangle, data)
        else:
            self._apply_subdivision(triangle, self.division_type, self.kwargs)
        if self.relabel:
            triangle.relabel_vertices()

        new_entry = Data(**data.to_dict())
        new_entry.triangulation = triangle.to_list()
        new_entry.n_vertices = triangle.n_vertices
        new_entry.id = f"{data.id}_{tag}"

        return new_entry

    def _build_ood_split(self, test_entries: List[Data], rng: random.Random):
        """Build the OOD split by subdividing the test-set entries."""

        # In the case where we don't want a subdivision just return the test set
        if self.division_type == SubdivisionType.NONE:
            return test_entries

        k = (
            self.max_ood_size_per_class
            if self.max_ood_size_per_class is not None
            else int(1e9)
        )

        # Construct class dict
        entries_by_class = defaultdict(list)
        for data in test_entries:
            entries_by_class[data.name].append(data)

        # All sources are drawn before any subdivision runs, so every
        # subdivision of one seed starts from the same source entries,
        # however many random draws the subdivisions themselves make.
        sources_by_class = {}
        for class_name in sorted(entries_by_class):
            # Choose only k samples  if specified with `k`,
            #  if there's less than k, choose the maximum amount of samples in a `class_name`
            k_cap: int = (
                min(len(entries_by_class[class_name]), k)
                if k
                else len(entries_by_class[class_name])
            )
            if k is not None and k_cap < k:
                warnings.warn(
                    f"Not enough samples of '{class_name}'"
                    "increase the size of test (split or amount of samples) "
                    "or lower the class count number"
                )
            sources_by_class[class_name] = rng.choices(
                entries_by_class[class_name], k=k_cap
            )

        ood_list: List = []
        for class_name, sources in sources_by_class.items():
            for i, source in enumerate(
                tqdm(sources, desc=f"Subdividing OOD ({class_name})")
            ):
                ood_list.append(self._subdivide_entry(source, rng, f"ood_{i}"))

        return ood_list

    def process(self):
        """Processes dataset."""
        inputs = self._load_raw_entries()
        rng = random.Random(self.seed)

        data_list = [Data(**el) for el in inputs]

        if self.pre_filter is not None:
            data_list = [
                data
                for data in tqdm(data_list, desc="Filtering")
                if self.pre_filter(data)
            ]
        # Make sure that max_vertices is enforced
        if self.max_vertices is not None:
            data_list = [
                data
                for data in data_list
                if data.n_vertices <= self.max_vertices
            ]

        if self.balanced:
            # balance_dataset enforces the vertex cap itself, both as a
            # prefilter and during augmentation.
            data_list = balance_dataset(
                data_list,
                seed=self.seed,
                max_vertices=self.max_vertices,
                target_count=self.target_count,
                n_moves=self.n_moves,
                use_surgery=self.use_surgery,
            )

        # Cap the vertex count of the in-distribution splits
        if self.division_type == SubdivisionType.GRADED:
            assert (
                max([d.n_vertices for d in data_list])
                < self.kwargs["graded_vertex_number"]
            ), "The dataset contains triangulations with more vertices than `graded_vertex_number`"

        # Filter by homeomorphism type
        data_list, _ = filter_by_class_count(
            data_list, "name", self.min_sample_per_class
        )

        # Get the class labels
        labels = (
            np.array([data.name for data in data_list])
            if self.stratified
            else None
        )

        # Make index splits
        train_index, val_index, test_index = make_split_index(
            data_list_size=len(data_list),
            seed=self.seed,
            train_size=self.split_proportions[0],
            val_size=self.split_proportions[1],
            test_size=self.split_proportions[2],
            labels=labels,
        )

        split_lists = {
            "train": [data_list[idx] for idx in train_index],
            "val": [data_list[idx] for idx in val_index],
            "test": [data_list[idx] for idx in test_index],
        }

        # Apply the selected subdivision algorithm to the test set
        split_lists["ood"] = self._build_ood_split(
            test_entries=split_lists["test"], rng=rng
        )

        # The OOD split derives from the unexpanded test entries, so
        # expansion happens after it has been built.
        for split_type in SPLIT_TYPES[:3]:
            split_lists[split_type] = self._expand_split(
                split_type, split_lists[split_type]
            )

        for i, split_type in enumerate(SPLIT_TYPES):
            data_split_list = split_lists[split_type]
            if self.pre_transform is not None:
                data_split_list = [
                    self.pre_transform(data)
                    for data in tqdm(
                        data_split_list,
                        desc=f"Pre-transforming ({split_type})",
                    )
                ]
            # WARN:  This is order specific!
            self.save(data_split_list, self.processed_paths[i])

    def _expand_split(self, split_type: str, data_list: List[Data]):
        """Hook for subclasses to expand a train, val or test split.

        Runs before the pre-transform; the base class returns the
        entries unchanged.
        """
        return data_list
