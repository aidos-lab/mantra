import json
import random
from enum import Enum
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import (
    Data,
)
from tqdm import tqdm

from mantra.augmentations import Triangulation
from mantra.datasets import ManifoldTriangulations
from mantra.datasets.utils import filter_by_class_count


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
    with subdivisions on the test set
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
        split_proportions: List[float] = [0.6, 0.2, 0.2],
        stratified=False,
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
        split_proportions : List[str]
            Proportional split in terms of [train, val, test]
        stratified : bool
            If to use stratified splitting.
        kwargs : Dict
            Arguments for the subdivision.
        """
        self.stratified = stratified
        self.split_type = split_type
        self.split_proportions = split_proportions
        self.class_count_filter = class_count_filter
        self.division_type = SubdivisionType.from_str(division_type)
        self.kwargs = kwargs

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

    def _build_ood_str(self):
        base_str = str(self.division_type)
        if self.division_type == SubdivisionType.BARYCENTRIC:
            arg_str = f"{self.kwargs.get('round', 1)}"
        elif self.division_type == SubdivisionType.STELLAR:
            arg_str = f"{self.kwargs.get('fraction', 1)}"
        else:  # Graded
            arg_str = f"{self.kwargs.get('min_vertices', 1)}"

        return base_str + f"_{arg_str}"

    @property
    def processed_file_names(self):
        """Return process file names.

        Stores the processed data in a file. If this file is present in the
        `processed` folder, processing will typically be skipped.
        """
        base_files = []
        for split_type in ["train", "val", "test"]:
            file_str = f"{split_type}.pt"
            base_files.append(file_str)

        ood_file: str = f"ood_{self._build_ood_str()}.pt"
        base_files.append(ood_file)

        return base_files

    def _calculate_induced_vertices(self):
        """ " This should calculate the number of vertices
        induced by the application of this subdivision

        """
        if self.division_type == SubdivisionType.BARYCENTRIC:
            return 4
        # TODO: This might change by dimension
        elif self.division_type == SubdivisionType.STELLAR:
            return 1
        else:  # self.division_type == SubdivisionType.GRADED:
            return 1

    def _subdivide_triangle(self, data, **kwargs):
        triangulation = data["triangulation"]
        rng = random.Random(self.seed)

        triangle = Triangulation(triangulation, rng)
        fn_str = f"{str(self.division_type)}_subdivision"
        fn = getattr(triangle, fn_str)

        new_triangulation, n_v = fn(kwargs)

        new_entry = triangulation.copy()
        new_entry["triangulation"] = new_triangulation
        new_entry["n_vertices"] = n_v

        return new_entry

    def _apply_subdivision(self, data_list):
        """Apply the subdivision to the data_list"""
        if self.division_type == SubdivisionType.BARYCENTRIC:
            rounds = self.kwargs.get("round", 1)
            for _ in range(rounds):
                data_list = [self.subdivide_triangle(tri) for tri in data_list]
        elif self.division_type == SubdivisionType.STELLAR:
            fraction = self.kwargs.get("fraction", 1)
            data_list = [
                self.subdivive_triangle(tri, fraction=fraction)
                for tri in data_list
            ]
        else:  # This is the graded
            min_vertices = self.kwargs.get("min_vertices", 1)
            data_list = [
                self.subdivide_triangle(tri, over_vrtx_cnt=min_vertices)
                for tri in data_list
            ]

        return data_list

    def process(self):
        """Processes dataset."""
        with open(self.raw_paths[0]) as f:
            inputs = json.load(f)

        data_list = [Data(**el) for el in inputs]

        if self.pre_filter is not None:
            data_list = [
                data
                for data in tqdm(data_list, desc="Filtering")
                if self.pre_filter(data)
            ]

        # Filter by homeomorphism type
        data_list, _ = filter_by_class_count(
            data_list, "name", self.class_count_filter
        )
        y_values = np.array([data.y for data in data_list])

        train_size, val_size, test_size = self.split_proportions
        # Train / test split
        train_val_index, test_index = train_test_split(
            np.arange(len(data_list)),
            test_size=test_size,
            shuffle=True,
            stratify=(y_values if self.stratified else None),
            random_state=self.seed,
        )

        # train val split
        train_index, val_index = train_test_split(
            train_val_index,
            test_size=val_size / (train_size + val_size),
            shuffle=True,
            stratify=(y_values[train_val_index] if self.stratified else None),
            random_state=self.seed,
        )

        # Apply the select subdivison algorithm
        data_test_list = [data_list[idx] for idx in test_index]

        ood_data_list = self._apply_subdivision(data_test_list)

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

        for i, split_type in enumerate(["train", "val", "test", "ood"]):
            data_split_list = [
                data_list[idx] for idx in split_dict[split_type]
            ]
            # WARN:  This is order specific!
            self.save(data_split_list, self.processed_paths[i])
