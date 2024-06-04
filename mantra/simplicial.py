"""
Simplical dataset, downloads the processed data from Zenodo into torch geometric 
dataset that can be used in conjunction to dataloaders. 

NOTE: Code untested until we have the zenodo database running or another place
retrieve the data from.

NOTE: Dowloading does not yet work until we have a public repository.
"""

import os
import json

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
    extract_gz,
)


class SimplicialDataset(InMemoryDataset):

    def __init__(
        self,
        root,
        manifold="2",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        if manifold not in ["2", "3"]:
            raise ValueError(
                f"Manifolds should either be 2 or 3, you provided {manifold}"
            )

        self.manifold = manifold
        root += f"/simplicial"
        # Note once we have an official release
        # we can set it to latest as default.
        self.release = "latest"
        self.url = f"https://github.com/aidos-lab/MANTRADataset/releases/download/{self.release}/{self.manifold}_manifolds.json.gz"
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


if __name__ == "__main__":
    dataset = SimplicialDataset(root="./data", manifold="3")
