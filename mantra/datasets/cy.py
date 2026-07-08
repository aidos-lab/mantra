import os
import shutil

import numpy as np
import pyarrow.parquet as pq
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from mantra.datasets import ManifoldTriangulations


class CY(InMemoryDataset):
    """Dataset of Calabi-Yau manifold triangulations."""

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
        debug=False,
    ):
        """
        Create a new CY-Manifolds dataset.

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
            If set, use a local parquet file instead of downloading from
            GitHub. The file will be copied into the raw directory.
            Useful for testing locally generated datasets.

        debug : bool
            Only load 1k triangulations for debug purposes.
        """
        self.version = version
        self.name = name
        self.local_path = os.path.abspath(local_path) if local_path else None
        self.debug = debug

        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload,
        )

    def _add_version_to_root(self):
        if self.version == "latest":
            return "/cy/debug/" if self.debug else "/cy/"
        else:
            return (
                f"/cy/{self.version}/debug/"
                if self.debug
                else f"/cy/{self.version}/"
            )

    @property
    def raw_file_names(self):
        """Return raw file names.

        Stores the raw file names that need to be present in the raw folder
        for downloading to be skipped. To reference raw file names, use the
        property `self.raw_paths`.
        """
        return ["manifolds.parquet"]

    @property
    def processed_file_names(self):
        """Return process file names.

        Stores the processed data in a file. If this file is present in the
        `processed` folder, processing will typically be skipped.
        """
        return ["data.pt"]

    def download(self):
        """Copy a local parquet file into the raw directory."""
        if self.local_path is None:
            raise NotADirectoryError("Downloading not implemented yet")
        dst = os.path.join(self.raw_dir, self.raw_file_names[0])
        shutil.copy2(self.local_path, dst)

    def process(self):
        """Processes dataset."""
        parquet_file = pq.ParquetFile(self.raw_paths[0])

        data_list = []

        for pq_batch in parquet_file.iter_batches(batch_size=1000):
            parquet_df = pq_batch.to_pandas()
            for _, row in parquet_df.iterrows():
                row_dict = row.to_dict()

                # Rename the column to match downstream processing.
                row_dict["triangulation"] = row_dict["simplices"]

                # Convert vertex coordinates to a float tensor.
                row_dict["vertices"] = np.vstack(row_dict["vertices"])
                row_dict["vertices"] = torch.as_tensor(
                    row_dict["vertices"], dtype=torch.float32
                )

                del row_dict["simplices"]

                data_list.append(
                    Data(
                        **row_dict,
                        dimension=len(row_dict["triangulation"][0]),
                    )
                )

            if self.debug:
                break

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
