"""Datasets module

This module contains datasets describing triangulations of manifolds,
following the API of `pytorch-geometric`.
"""

import json
import os
import requests

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)


def get_url(version: str, manifold: str) -> str:
    """
    Function to fetch the full download url.
    """
    if version == "latest":
        return f"https://github.com/aidos-lab/MANTRA/releases/latest/download/{manifold}_manifolds.json.gz"  # noqa

    # Set headers for GitHub API
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Fetch the list of versions.
    response = requests.get(
        "https://api.github.com/repos/aidos-lab/mantra/releases",
        headers=headers,
    )
    all_available_versions = [item["name"] for item in response.json()]
    if version not in all_available_versions:
        raise ValueError(
            f"Version {version} not available, please choose one of the following versions: {all_available_versions}."
        )

    # Note that the url order is different (inconsistent) for a
    # specific release as compared to "latest".
    return f"https://github.com/aidos-lab/MANTRA/releases/download/{version}/{manifold}_manifolds.json.gz"  # noqa


class ManifoldTriangulations(InMemoryDataset):

    def __init__(
        self,
        root,
        manifold="2",
        version="latest",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        """
        The dataset class for the manifold triangulations.

        Parameters
        ----------
        manifold: string
            Wether to use the 2- or 3- manifolds. The 2-manifold dataset
            consist of all surfaces with up to 10 vertices. The
            3-manifolds consist of all triangulatoins with up to 10 vertices.
        version: string
            Version of the dataset to use. The version should correspond to a
            released version of the dataset, all of which can be found
            `on GitHub <https://github.com/aidos-lab/mantra/releases>`__.
            By default, the latest version will be downloaded. Unless
            specific reproducibility requirements are to be met, using
            `latest` is recommended.
        """
        # The properties need to be set before the super().__init__() call to
        # make sure they exist during processing. The process and download are
        # called during the super call.

        if manifold not in ["2", "3"]:
            raise ValueError(
                f"Manifolds should either be 2 or 3, you provided {manifold}"
            )

        self.manifold = manifold
        self.version = version
        self.url = get_url(version, manifold)

        if version == "latest":
            root += f"/mantra/{self.manifold}D"
        else:
            root += f"/mantra/{version}/{self.manifold}D"

        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload
        )

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """
        Stores the raw  file names that need to be present in the raw folder for
        downloading to be skipped. To reference raw file names, use the property
        self.raw_paths.
        """
        return [
            f"{self.manifold}_manifolds.json",
        ]

    @property
    def processed_file_names(self):
        """
        Stores the processed data in a file, if this file is present in the
        processed folder, it will skip processing. Othewise it will run the
        process function.
        """
        return [f"data_{self.manifold}.pt"]

    def download(self) -> None:
        """
        Downloads the specified version of the 2 or 3 manifolds in json format
        into the raw folder and extracts the results. The dataset version can
        specified when instantiating the class.
        """
        path = download_url(self.url, self.raw_dir)
        extract_gz(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        """
        Processes the raw json file and loads the result into a torch-geometric
        dataset. If provided during initialization, pretransforms and/or
        prefilters are applied before saving the preprocessed dataset.
        More information on pretransforms and prefilters can be found in the
        `pytorch-geometric documentation <https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html>`__.
        """
        with open(self.raw_paths[0]) as f:
            inputs = json.load(f)

        data_list = [Data(**el) for el in inputs]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
