"""Datasets module

This module contains datasets describing triangulations of manifolds,
following the API of `pytorch-geometric`.
"""

import abc
import json
import os

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)
from tqdm import tqdm


class ManifoldTriangulations(InMemoryDataset, abc.ABC):
    """Base class for storing manifold triangulations."""

    def __init__(
        self,
        root,
        version="latest",
        name=None,
        local_path=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        """
        Create a new dataset of manifold triangulations.

        Parameters
        ----------
        version : str
            Version of the dataset to use. The version should correspond to a
            released version of the dataset, all of which can be found
            `on GitHub <https://github.com/aidos-lab/mantra/releases>`__.
            By default, the latest version will be downloaded. Unless
            specific reproducibility requirements are to be met, using
            `latest` is recommended.

        name : str or None
            If set, the name denotes a way to distinguish between datasets
            based on the *same* data source but potentially prepared in a
            different fashion, i.e., by a different set of pre-transforms.

            Using different names enables such datasets to coexist in
            parallel. Otherwise, the `force_reload` flag of the base class
            has to be used always, obviating the need for pre-processing the
            dataset. The name will be used for storing all processed files of
            the dataset. Under the hood, this property will only change the
            place in which processed data is stored.

            As a suggestion the name should not include any spaces, thus
            making it easier to parse for the OS.

        local_path : str or None
            If set, use a local JSON file instead of downloading from
            GitHub. The file will be copied into the raw directory.
            Useful for testing locally generated datasets.
        """
        self.name = name
        self.version = version
        self.local_path = os.path.abspath(local_path) if local_path else None

        root += self._add_version_to_root()

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

        self.load(self.processed_paths[0])

    @abc.abstractmethod
    def _add_version_to_root(self):
        """Return the dataset-specific suffix appended to the root path."""

    @property
    def processed_dir(self):
        """Return directory for storing processed data."""
        if self.name is not None:
            return os.path.join(self.root, "processed", self.name)
        else:
            return super().processed_dir

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

        if self.pre_transform is not None:
            data_list = [
                self.pre_transform(data)
                for data in tqdm(data_list, desc="Pre-transforming")
            ]

        self.save(data_list, self.processed_paths[0])
