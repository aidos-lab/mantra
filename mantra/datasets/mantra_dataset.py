import glob
import math
import os
import random
import shutil
import warnings
from collections import defaultdict
from enum import Enum
from typing import List

import numpy as np
import torch
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

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(sub_name: str):
        for sub in SubdivisionType:
            if str(sub).lower() == sub_name.lower():
                return sub
        raise ValueError(f"There is no Subdivision with name {sub_name}")


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
            barycentric, graded, stellar.
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
            If set, subdivide at most this many distinct test-set
            sources per class, drawn uniformly at random; classes with
            fewer test entries contribute all of them.
        kwargs : Dict
            Arguments for the subdivision. Barycentric accepts ``round``
            (number of rounds, default 1), stellar accepts ``fraction``
            (fraction of top-simplices to subdivide, default 1.0), and
            graded requires ``graded_vertex_number``: every OOD sample is grown
            to exactly this number of vertices, and test-set sources that
            already have ``graded_vertex_number`` or more vertices are excluded
            from the OOD split.
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
        self.kwargs = kwargs

        if balanced and min_sample_per_class:
            warnings.warn(
                "balanced=True equalizes the classes before "
                "min_sample_per_class is applied, so the filter can "
                "re-imbalance or drop classes again."
            )
        if self.division_type == SubdivisionType.GRADED:
            if "graded_vertex_number" not in kwargs:
                raise ValueError(
                    "Graded subdivision requires a 'graded_vertex_number' keyword "
                    "argument: the number of vertices every OOD sample is "
                    "grown to."
                )
            if (
                max_vertices is not None
                and kwargs["graded_vertex_number"] <= max_vertices
            ):
                raise ValueError(
                    f"graded_vertex_number ({kwargs['graded_vertex_number']}) must be "
                    f"strictly greater than max_vertices ({max_vertices}); "
                    "otherwise OOD samples are not guaranteed to be larger "
                    "than the train/val/test triangulations."
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

    def _build_ood_str(self):
        base_str = str(self.division_type)

        if self.division_type == SubdivisionType.NONE:
            return base_str

        elif self.division_type == SubdivisionType.BARYCENTRIC:
            arg_str = f"{self.kwargs.get('round', 1)}"
        elif self.division_type == SubdivisionType.STELLAR:
            arg_str = f"{self.kwargs.get('fraction', 1)}"
        else:  # self.division_type == SubdivisionType.GRADED:  # Graded
            arg_str = f"{self.kwargs['graded_vertex_number']}"

        ood_str = base_str + f"_{arg_str}"

        if self.max_ood_size_per_class is not None:
            ood_str += f"_cap{self.max_ood_size_per_class}"

        return ood_str

    @property
    def processed_file_names(self):
        """Return process file names.

        The train/val/test files and the raw test entries (``sources``)
        are shared by every subdivision of one split; each subdivision
        adds its own ``ood_*`` file.
        """
        suffix = self._split_file_suffix()
        names = [f"{split_type}{suffix}.pt" for split_type in SPLIT_TYPES[:3]]
        names.append(f"ood_{self._build_ood_str()}{suffix}.pt")
        names.append(f"sources{suffix}.pt")
        return names

    def _subdivide_entry(self, data, rng, tag):
        """Return a copy of ``data`` with the subdivided triangulation."""
        triangle = Triangulation.from_list(data.triangulation, rng=rng)

        if self.division_type == SubdivisionType.BARYCENTRIC:
            for _ in range(self.kwargs.get("round", 1)):
                triangle.barycentric_subdivision()
        elif self.division_type == SubdivisionType.STELLAR:
            triangle.stellar_subdivision(
                fraction=self.kwargs.get("fraction", 1.0)
            )
        else:  # Graded
            triangle.graded_subdivision(
                over_vrtx_cnt=self.kwargs["graded_vertex_number"]
            )

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

        ood_list: List = []
        for class_name in sorted(entries_by_class):
            # Choose only k samples  if specified with `k`,
            #  if there's less than k, choose the maximum amount of samples in a `class_name`
            pool = entries_by_class[class_name]
            k_cap = min(len(pool), k)
            assert self.max_ood_size_per_class is None or k_cap >= k, (
                f"Class '{class_name}' has {k_cap} test entries, fewer "
                f"than max_ood_size_per_class={k}. Enlarge the test split or lower the cap "
                "for an OOD split with equal class sizes."
            )

            sources = rng.sample(pool, k_cap)
            for i in tqdm(
                range(len(sources)), desc=f"Subdividing OOD ({class_name})"
            ):
                source = sources[i]
                ood_list.append(self._subdivide_entry(source, rng, f"ood_{i}"))

        return ood_list

    def process(self):
        """Build the processed files that are missing.

        The train/val/test files and the raw test entries are built once
        per processed directory. A further subdivision only subdivides
        the cached test entries and pre-transforms its own OOD split, so
        the in-distribution files are never rewritten. Rebuilding the
        splits deletes the OOD files derived from the previous test
        entries, so ``force_reload`` on one split refreshes them all.
        """
        # The [4:] is just to return a list it's not that there are more elements!
        # We check everything but the OOD-file at position 3
        shared = self.processed_paths[:3] + self.processed_paths[4:]
        if self.force_reload or not all(os.path.exists(p) for p in shared):
            suffix = self._split_file_suffix()
            for path in glob.glob(
                os.path.join(self.processed_dir, f"ood_*{suffix}.pt")
            ):
                os.remove(path)
            self._process_splits()
        self._process_ood()

    def _process_ood(self):
        """Build the OOD file from the cached raw test entries."""
        if self.division_type == SubdivisionType.NONE:
            shutil.copyfile(self.processed_paths[2], self.processed_paths[3])
            return

        sources = torch.load(self.processed_paths[4])
        if (
            self.division_type == SubdivisionType.GRADED
            and sources["max_n_vertices"]
            >= self.kwargs["graded_vertex_number"]
        ):
            raise ValueError(
                "The dataset contains triangulations with at least "
                "`graded_vertex_number` vertices"
            )

        ood_data_list = self._build_ood_split(
            test_entries=[Data(**el) for el in sources["entries"]],
            rng=random.Random(self.seed),
        )
        if self.pre_transform is not None:
            ood_data_list = [
                self.pre_transform(data)
                for data in tqdm(ood_data_list, desc="Pre-transforming")
            ]
        self.save(ood_data_list, self.processed_paths[3])

    def _process_splits(self):
        """Build the train/val/test files and the raw test entries."""
        inputs = self._load_raw_entries()

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
            data_list = balance_dataset(
                data_list,
                seed=self.seed,
                max_vertices=self.max_vertices,
                target_count=self.target_count,
                n_moves=self.n_moves,
                use_surgery=self.use_surgery,
            )

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

        # The raw test entries seed every subdivision of this split
        sources = {
            "entries": [data_list[idx].to_dict() for idx in test_index],
            "max_n_vertices": max(int(d.n_vertices) for d in data_list),
        }

        torch.save(sources, self.processed_paths[4])

        if self.pre_transform is not None:
            data_list = [
                self.pre_transform(data)
                for data in tqdm(data_list, desc="Pre-transforming")
            ]

        for path, index in zip(
            self.processed_paths, [train_index, val_index, test_index]
        ):
            self.save([data_list[idx] for idx in index], path)
