import math
from typing import List

import torch
from tqdm import tqdm

from mantra.datasets.calabi_yau import CalabiYau
from mantra.datasets.utils import make_split_index

SPLIT_TYPES = ["train", "val", "test"]
DEFAULT_SPLIT_PROPORTIONS = [0.6, 0.2, 0.2]


class CalabiYauDataset(CalabiYau):
    """Train/val/test split of the :class:`CalabiYau` dataset.

    Mirrors :class:`~mantra.datasets.MantraDataset`: the full data is
    split once into seeded, deterministic train/val/test parts and each
    instance serves the part selected by ``split_type``. The split
    files live next to the full ``data.pt`` of the base class and their
    names encode every option that changes the splits, so differently
    seeded, proportioned, filtered or stratified variants coexist.
    """

    def __init__(
        self,
        root,
        split_type: str,
        version="latest",
        name=None,
        local_path=None,
        limit=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
        seed: int = 42,
        split_proportions: List[float] = DEFAULT_SPLIT_PROPORTIONS,
        stratified: bool = False,
        label_source: str = "h11",
        parquet_batch_size: int = 1000,
    ):
        """
        Create a split of the CY-Manifolds dataset.

        Parameters
        ----------
        split_type : str
            One of ``train``, ``val`` or ``test``.

        seed : int
            Seed of the split assignment.

        split_proportions : List[float]
            Proportional split in terms of [train, val, test]. Must sum
            to 1.

        stratified : bool
            If to use stratified splitting by the values of
            ``label_source``. Every value then needs enough samples to
            appear in each split, so combine it with
            ``min_sample_per_class``.

        label_source : str
            Attribute whose values define the classes used by
            ``stratified`` and ``min_sample_per_class``, e.g. ``h11``.

        min_sample_per_class : int or None
            Drop the samples whose ``label_source`` value occurs at most
            this many times before splitting. ``None`` keeps every
            sample.

        The remaining parameters are those of :class:`CalabiYau`.
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

        self.split_type = split_type
        self.seed = seed
        self.split_proportions = split_proportions
        self.stratified = stratified
        self.label_source = label_source

        super().__init__(
            root,
            version=version,
            name=name,
            local_path=local_path,
            limit=limit,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
            parquet_batch_size=parquet_batch_size,
        )

    def _load_index(self):
        """Load the processed file matching ``split_type``."""
        return SPLIT_TYPES.index(self.split_type)

    def _split_file_suffix(self):
        """Suffix encoding the options that change the splits.

        The processed directory of :class:`CalabiYau` carries no seed
        (unlike :class:`ManifoldTriangulations`), so the seed is part
        of the file names.
        """
        parts = [f"seed{self.seed}"]
        if self.split_proportions != DEFAULT_SPLIT_PROPORTIONS:
            parts.append(
                "sp" + "-".join(str(p) for p in self.split_proportions)
            )
        if self.stratified:
            parts.append("strat")
            parts.append(self.label_source)
        return "_" + "_".join(parts)

    @property
    def processed_file_names(self):
        """Return process file names, one per split."""
        suffix = self._split_file_suffix()
        return [f"{split_type}{suffix}.pt" for split_type in SPLIT_TYPES]

    def process(self):
        """Split the parsed data and save one file per split."""
        data_list = self._load_data_list()

        if self.pre_filter is not None:
            data_list = [
                data
                for data in tqdm(data_list, desc="Filtering")
                if self.pre_filter(data)
            ]

        labels = (
            torch.tensor([data[self.label_source] for data in data_list])
            if self.stratified
            else None
        )

        if self.pre_transform is not None:
            data_list = [
                self.pre_transform(data)
                for data in tqdm(data_list, desc="Pre-transforming")
            ]

        train_index, val_index, test_index = make_split_index(
            data_list_size=len(data_list),
            seed=self.seed,
            train_size=self.split_proportions[0],
            val_size=self.split_proportions[1],
            test_size=self.split_proportions[2],
            labels=labels,
        )

        # WARN: This is order specific (must match SPLIT_TYPES).
        for path, index in zip(
            self.processed_paths, (train_index, val_index, test_index)
        ):
            self.save([data_list[idx] for idx in index], path)
