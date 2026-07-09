"""Datasets module
This module contains datasets describing triangulations of manifolds,
following the API of `pytorch-geometric`.
"""

import json
import os
import shutil

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)
from tqdm import tqdm

from mantra.datasets.utils import _get_mantra_dataset_url


class ManifoldTriangulations(InMemoryDataset):
    """Base class for storing manifold triangulations."""

    def __init__(
        self,
        root,
        version="latest",
        dimension=2,
        name=None,
        balanced=False,
        local_path=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
        seed=42,
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
        dimension : int
            Dimension of manifold triangulations to load. Currently, only
            2 or 3 are supported, denoting 2-manifolds (i.e., surfaces)
            and 3-manifolds, respectively.
        balanced : bool
            If True, download the balanced variant of the dataset.
            Balanced datasets have been augmented via Pachner moves
            so that all manifold classes have roughly equal
            representation and vertex count distributions.
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
        seed : int
            Seed for generating additional triangulations or augmentations.
        """
        assert dimension in [2, 3], "Dimension can only be 2 or 3"
        self.version = version
        self.seed = seed
        self.balanced = balanced
        self.name = name
        self.dimension = dimension
        self.version = version
        self.local_path = os.path.abspath(local_path) if local_path else None
        self.url = _get_mantra_dataset_url(version, dimension, balanced)

        root += self._add_version_to_root()

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

        self.load(self.processed_paths[self._load_index()])

    def _load_index(self):
        """Index into ``processed_paths`` of the file to load.

        Subclasses producing several processed files (e.g. one per
        split) override this to select the right one.
        """
        return 0

    def _add_version_to_root(self):
        if self.version == "latest":
            return f"/mantra/{self.dimension}D"
        else:
            return f"/mantra/{self.version}/{self.dimension}D"

    @property
    def raw_file_names(self):
        """Return raw file names.

        Stores the raw file names that need to be present in the raw folder
        for downloading to be skipped. To reference raw file names, use the
        property `self.raw_paths`.
        """
        return [f"{self.dimension}_manifolds.json"]

    @property
    def processed_dir(self):
        """Return directory for storing processed data."""
        base_path = os.path.join(self.root, "processed")
        balanced_suffix = "balanced" if self.balanced else "unbalanced"
        balanced_suffix = f"{balanced_suffix}_{self.seed}"

        if self.name is not None:
            base_path = os.path.join(base_path, self.name)

        base_path = os.path.join(base_path, balanced_suffix)

        return base_path

    @property
    def processed_file_names(self):
        """Return process file names.

        Stores the processed data in a file. If this file is present in the
        `processed` folder, processing will typically be skipped.
        """
        return ["full.pt"]

    def download(self):
        """Download dataset depending on specified version."""
        if self.local_path is not None:
            dst = os.path.join(self.raw_dir, self.raw_file_names[0])
            shutil.copy2(self.local_path, dst)
        else:
            path = download_url(self.url, self.raw_dir)
            extract_gz(path, self.raw_dir)
            os.unlink(path)

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
