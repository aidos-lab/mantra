import json
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

from mantra.augmentations import Triangulation
from mantra.datasets.mantra import ManifoldTriangulations
from mantra.datasets.utils import filter_by_class_count, make_split_index

SPLIT_TYPES = ["train", "val", "test", "ood"]
DEFAULT_SPLIT_PROPORTIONS = [0.6, 0.2, 0.2]


class SubdivisionType(Enum):
    STELLAR = 1
    GRADED = 2
    BARYCENTRIC = 3

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(sub_name: str):
        for sub in SubdivisionType:
            if str(sub) == sub_name:
                return sub
        raise ValueError(f"There is no Subdivision with name {sub_name}")


class MANTRADivided(ManifoldTriangulations):
    """Dataset of manifold triangulations from the MANTRA benchmark
    with subdivisions of the test set as an additional OOD split.
    """

    def __init__(
        self,
        root,
        split_type: str,
        dimension=2,
        version="latest",
        balanced=False,
        name=None,
        local_path=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
        seed=42,
        division_type: str = "barycentric",
        class_count_filter=None,
        split_proportions: List[float] = DEFAULT_SPLIT_PROPORTIONS,
        stratified=False,
        max_vertices=None,
        max_ood_size_per_class=None,
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
        class_count_filter : int or None
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
            sample, since ``vertex_number`` must exceed ``max_vertices``.
        max_ood_size_per_class : int or None
            If set, oversample and trim the OOD split so that every
            class contains exactly this many samples (classes without
            eligible test-set sources are skipped). Oversampling draws
            additional randomized subdivisions from the same test-set
            sources; it only yields distinct triangulations for
            randomized subdivisions (graded, or stellar with
            ``fraction`` < 1), while barycentric subdivision is
            deterministic and produces exact duplicates.
        kwargs : Dict
            Arguments for the subdivision. Barycentric accepts ``round``
            (number of rounds, default 1), stellar accepts ``fraction``
            (fraction of top-simplices to subdivide, default 1.0), and
            graded requires ``vertex_number``: every OOD sample is grown
            to exactly this number of vertices, and test-set sources that
            already have ``vertex_number`` or more vertices are excluded
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
        self.class_count_filter = class_count_filter
        self.division_type = SubdivisionType.from_str(division_type)
        self.max_vertices = max_vertices
        self.max_ood_size_per_class = max_ood_size_per_class
        self.kwargs = kwargs

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
                    f"graded_vertex_number ({kwargs['vertex_number']}) must be "
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
        )

    def _load_index(self):
        """Load the processed file matching ``split_type``."""
        return SPLIT_TYPES.index(self.split_type)

    def _split_file_suffix(self):
        """Suffix encoding parameters that change the train/val/test data."""
        parts = []
        if self.max_vertices is not None:
            parts.append(f"mv{self.max_vertices}")
        if self.class_count_filter:
            parts.append(f"ccf{self.class_count_filter}")
        if self.split_proportions != DEFAULT_SPLIT_PROPORTIONS:
            parts.append(
                "sp" + "-".join(str(p) for p in self.split_proportions)
            )
        if self.stratified:
            parts.append("strat")
        return "_" + "_".join(parts) if parts else ""

    def _build_ood_str(self):
        base_str = str(self.division_type)
        if self.division_type == SubdivisionType.BARYCENTRIC:
            arg_str = f"{self.kwargs.get('round', 1)}"
        elif self.division_type == SubdivisionType.STELLAR:
            arg_str = f"{self.kwargs.get('fraction', 1)}"
        else:  # Graded
            arg_str = f"{self.kwargs['vertex_number']}"

        ood_str = base_str + f"_{arg_str}"
        if self.max_ood_size_per_class is not None:
            ood_str += f"_cap{self.max_ood_size_per_class}"

        return ood_str

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
        k = self.max_ood_size_per_class

        # Construct class dict
        entries_by_class = defaultdict(list)
        for data in test_entries:
            entries_by_class[data.name].append(data)

        ood_list: List = []
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
            sources = rng.choices(entries_by_class[class_name], k=k_cap)

            for i in tqdm(
                range(len(sources)), desc=f"Subdividing OOD ({class_name})"
            ):
                source = sources[i]
                ood_list.append(self._subdivide_entry(source, rng, f"ood_{i}"))

        return ood_list

    def process(self):
        """Processes dataset."""
        rng = random.Random(self.seed)

        with open(self.raw_paths[0]) as f:
            inputs = json.load(f)

        data_list = [Data(**el) for el in inputs]

        if self.pre_filter is not None:
            data_list = [
                data
                for data in tqdm(data_list, desc="Filtering")
                if self.pre_filter(data)
            ]

        # Cap the vertex count of the in-distribution splits
        if self.division_type == SubdivisionType.GRADED:
            assert max([d.n_vertices for d in data_list]) < self.kwargs.get(
                "graded_vertex_number"
            ), "The dataset contains triangulations with more vertices than `graded_vertex_number`"

        # Filter by homeomorphism type
        data_list, _ = filter_by_class_count(
            data_list, "name", self.class_count_filter
        )

        # Get the class labels
        labels = np.array([data.name for data in data_list])

        # Make index splits
        train_index, val_index, test_index = make_split_index(
            data_list_size=len(data_list),
            seed=self.seed,
            train_size=self.split_proportions[0],
            val_size=self.split_proportions[1],
            test_size=self.split_proportions[2],
            labels=labels,
        )

        # Apply the selected subdivision algorithm to the test set
        ood_data_list = self._build_ood_split(
            test_entries=[data_list[idx] for idx in test_index], rng=rng
        )

        # Get the indices for ood
        ood_index = np.arange(
            len(data_list), len(data_list) + len(ood_data_list)
        )

        # Dictionary with splits
        split_dict = {
            "train": train_index,
            "val": val_index,
            "test": test_index,
            "ood": ood_index,
        }

        # Stick it at the end
        data_list.extend(ood_data_list)

        # Apply pretransforms now
        if self.pre_transform is not None:
            data_list = [
                self.pre_transform(data)
                for data in tqdm(data_list, desc="Pre-transforming")
            ]

        for i, split_type in enumerate(SPLIT_TYPES):
            data_split_list = [
                data_list[idx] for idx in split_dict[split_type]
            ]
            # WARN:  This is order specific!
            self.save(data_split_list, self.processed_paths[i])
