"""Datasets module

This module contains datasets describing triangulations of manifolds,
following the API of `pytorch-geometric`.
"""

import json
import os
import requests

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import download_url
from torch_geometric.data import extract_gz


def _get_dataset_url(version: str, manifold: str) -> str:
    """Get URL to download dataset from."""
    if version == "latest":
        return f"https://github.com/aidos-lab/MANTRA/releases/latest/download/{manifold}_manifolds.json.gz"  # noqa

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(
        "https://api.github.com/repos/aidos-lab/mantra/releases",
        headers=headers,
    )

    all_available_versions = [item["name"] for item in response.json()]

    if version not in all_available_versions:
        raise ValueError(
            f"Version {version} not available, please choose one of the following versions: {all_available_versions}."  # noqa
        )

    # Note that the URL order is different and thus inconsistent for a
    # specific release.
    return f"https://github.com/aidos-lab/MANTRA/releases/download/{version}/{manifold}_manifolds.json.gz"  # noqa


class ManifoldTriangulations(InMemoryDataset):
    """Base class for storing manifold triangulations."""

    def __init__(
        self,
        root,
        manifold="2",
        version="latest",
        name=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        """
        Create a new dataset of manifold triangulations.

        Parameters
        ----------
        manifold : str
            Dimension of manifolds to load. Currently, only "2" or "3"
            are supported, denoting surfaces and 3-manifolds,
            respectively.

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
            the dataset.
        """
        assert manifold in ["2", "3"]

        self.manifold = manifold
        self.name = name
        self.version = version
        self.url = _get_dataset_url(version, manifold)

        if version == "latest":
            root += f"/mantra/{self.manifold}D"
        else:
            root += f"/mantra/{version}/{self.manifold}D"

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """Return raw file names.

        Stores the raw file names that need to be present in the raw folder
        for downloading to be skipped. To reference raw file names, use the
        property `self.raw_paths`.
        """
        return [f"{self.manifold}_manifolds.json"]

    @property
    def processed_file_names(self):
        """Return process file names.

        Stores the processed data in a file. If this file is present in the
        `processed` folder, processing will typically be skipped.
        """
        if self.name is not None:
            return [f"data_{self.manifold}_{self.name}.pt"]
        else:
            return [f"data_{self.manifold}.pt"]

    def download(self) -> None:
        """Download dataset depending on specified version."""
        path = download_url(self.url, self.raw_dir)
        extract_gz(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        """Processes dataset."""
        with open(self.raw_paths[0]) as f:
            inputs = json.load(f)

        data_list = [Data(**el) for el in inputs]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
