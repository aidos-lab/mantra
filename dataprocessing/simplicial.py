"""
Simplical dataset, downloads the processed data from Zenodo into torch geometric 
dataset that can be used in conjunction to dataloaders. 

NOTE: Code untested until we have the zenodo database running or another place
retrieve the data from.
"""

import os
import json

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class SimplicialDataset(InMemoryDataset):
    url = "my.cool.url.com"

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.manifold = "2"
        root += f"/simplicial"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            f"{self.manifold}_manifold.json",
        ]

    @property
    def processed_file_names(self):
        return [f"{self.manifold}_manifold.pt"]

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):

        # After download and extraction, we expect a json file with
        # a list of json objects.
        inputs = json.load(self.raw_paths[0])

        data_list = [Data(**el) for el in inputs]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
