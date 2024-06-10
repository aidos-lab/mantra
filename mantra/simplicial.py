"""
The dataset class for the manifolds. It consists of 2 and 3 manifolds 
along with the topological information. We follow the pytorch geometric 
conventions for the dataset. 
"""

import os
import json

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)


class SimplicialDataset(InMemoryDataset):

    def __init__(
        self,
        root,
        manifold="2",
        version="latest",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        The dataset class for the manifold triangulations.

        Parameters
        ----------
        manifold: string
            Wether to use the 2 or 3 manifolds. Default is 2.
        version: string
            Version of the dataset to use. The version should correspond to
            a released version of the dataset which can be found
            ![here](https://github.com/aidos-lab/mantra/releases). Default is
            the latest version.
        """

        if manifold not in ["2", "3"]:
            raise ValueError(
                f"Manifolds should either be 2 or 3, you provided {manifold}"
            )

        self.manifold = manifold
        root += f"/simplicial"
        self.version = version
        self.url = f"https://github.com/aidos-lab/MANTRADataset/releases/{self.version}/download/{self.manifold}_manifolds.json.gz"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            f"{self.manifold}_manifolds.json",
        ]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_gz(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        with open(self.raw_paths[0]) as f:
            inputs = json.load(f)

        data_list = [Data(**el) for el in inputs]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
