import os
import shutil
from typing import List

import numpy as np
import pyarrow.parquet as pq
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


class CalabiYau(InMemoryDataset):
    """Dataset of Calabi-Yau manifold triangulations.

    Each sample is a fine star triangulation of a 4-dimensional
    reflexive lattice polytope, stored in a parquet file with columns
    `simplices` (top-level simplices, 0-indexed vertex lists) and
    `vertices` (integer lattice coordinates, one row per vertex). All
    remaining columns are attached to the resulting
    :class:`~torch_geometric.data.Data` objects: scalar columns (e.g.
    the Hodge numbers `h11`, `h12`) as Python scalars, list columns
    (e.g. the per-vertex second Chern class `c2` or the sparse
    `intersection_numbers` block) as tensors, so that they collate and
    cache like any other per-sample attribute.

    To stay compatible with the MANTRA conventions used by the
    transforms and representations of this library, `process()` converts
    the simplices to 1-indexed lists, sorted within each simplex and
    lexicographically across simplices, and stores the *topological*
    dimension of the complex (number of vertices per top simplex minus
    one) in the `dimension` attribute.

    This class serves the full dataset; train/val/test splits are
    served by :class:`~mantra.datasets.CalabiYauDataset`.
    """

    def __init__(
        self,
        root,
        version="latest",
        name=None,
        local_path=None,
        limit=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
        parquet_batch_size: int = 1000,
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

        limit : int or None
            Only process the first `limit` triangulations. Limited
            subsets are stored in their own processed directory, so
            they can coexist with the full dataset and are cheap to
            precompute, e.g. for smoke tests or timing benchmarks.

        parquet_batch_size : int
            Size of the batch to load from the parquet file of the
            manifold triangulations.
        """
        self.version = version
        self.name = name
        self.local_path = os.path.abspath(local_path) if local_path else None
        self.limit = limit
        self.parquet_batch_size = parquet_batch_size

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
        return int(0)

    def _add_version_to_root(self):
        if self.version == "latest":
            return "/calabi_yau/"
        else:
            return f"/calabi_yau/{self.version}/"

    @property
    def raw_file_names(self):
        """Return raw file names.

        Stores the raw file names that need to be present in the raw folder
        for downloading to be skipped. To reference raw file names, use the
        property `self.raw_paths`.
        """
        return ["manifolds.parquet"]

    @property
    def processed_dir(self):
        """Return path of directory with the processed files.

        The path encodes the optional `name` and `limit` parameters so
        that differently prepared variants of the dataset can coexist.
        """
        path = os.path.join(self.root, "processed")

        if self.name is not None:
            path = os.path.join(path, self.name)
        if self.limit is not None:
            path = os.path.join(path, f"limit_{self.limit}")

        return path

    @property
    def processed_file_names(self):
        """Return process file names.

        Stores the processed data in a file. If this file is present in
        the `processed` folder, processing will typically be skipped.
        """
        return ["data.pt"]

    def download(self):
        """Copy a local parquet file into the raw directory."""
        if self.local_path is None:
            raise NotADirectoryError("Downloading not implemented yet")
        dst = os.path.join(self.raw_dir, self.raw_file_names[0])
        shutil.copy2(self.local_path, dst)

    @staticmethod
    def _list_column_to_tensor(value):
        """Convert the entry of a parquet list column to a tensor.

        Plain ``list<T>`` columns arrive from pandas as 1-D numpy
        arrays. Nested ``list<list<T>>`` columns (e.g. the ``(nnz, 4)``
        COO block of ``intersection_numbers``) arrive as object arrays
        of arrays, which ``torch.as_tensor`` rejects; their rows are
        stacked explicitly. Ragged rows have no tensor form and raise
        instead of being silently mangled. Floating point values are
        stored as ``float32``.
        """
        array = np.asarray(value)

        if array.dtype == object:
            rows = [np.asarray(row) for row in value]
            if not rows:
                array = np.zeros((0,), dtype=np.int64)
            else:
                shapes = {row.shape for row in rows}
                if len(shapes) != 1:
                    raise ValueError(
                        "Cannot convert a list column with ragged rows "
                        f"(shapes {sorted(shapes)}) to a tensor"
                    )
                array = np.stack(rows)

        # `torch.tensor` copies: the arrays handed out by pyarrow are
        # read-only, which `as_tensor` would carry over with a warning.
        tensor = torch.tensor(array)
        if tensor.is_floating_point():
            tensor = tensor.to(torch.float32)
        return tensor

    @classmethod
    def _to_attribute(cls, value):
        """Convert a parquet cell to a `Data` attribute.

        Numpy scalars (e.g. labels like `h11`) become Python scalars so
        that they collate like the fields of the JSON-based MANTRA
        datasets; list columns become tensors. Everything else (e.g.
        strings) is attached as is.
        """
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (np.ndarray, list, tuple)):
            return cls._list_column_to_tensor(value)
        return value

    def _process_parquet(self, parquet_file) -> List[Data]:
        """Processes the parquet file iteration process.

        Parameters
        ----------
        parquet_file : ParquetFile
            Parquet file containing the dataset to parse.

        Returns
        ---------
        List[Data]
            List of Data object containing the triangulations.

        """
        data_list = []

        for pq_batch in parquet_file.iter_batches(
            batch_size=self.parquet_batch_size
        ):
            parquet_df = pq_batch.to_pandas()
            for _, row in parquet_df.iterrows():
                row_dict = row.to_dict()

                # Convert to the MANTRA convention: plain (nested) lists
                # of 1-indexed vertices, named `triangulation`. The
                # vertices of each simplex are sorted and the simplices
                # are ordered lexicographically, since the
                # representations and the feature propagation derive
                # faces from the stored order.
                triangulation = sorted(
                    sorted(int(v) + 1 for v in simplex)
                    for simplex in row_dict.pop("simplices")
                )

                # Convert vertex coordinates to a float tensor; row `i`
                # holds the coordinates of vertex `i + 1`.
                coords = torch.as_tensor(
                    np.vstack(row_dict.pop("vertices")), dtype=torch.float32
                )

                used_vertices = {v for s in triangulation for v in s}

                assert used_vertices == set(
                    range(1, coords.shape[0] + 1)
                ), "Not all vertices in the triangulation have a coordinate"

                attributes = {
                    k: self._to_attribute(v) for k, v in row_dict.items()
                }

                data_list.append(
                    Data(
                        triangulation=triangulation,
                        coords=coords,
                        dimension=len(triangulation[0]) - 1,
                        n_vertices=coords.shape[
                            0
                        ],  # We checked this was the number of vertices, i.e. len(used_vertices)
                        **attributes,
                    )
                )

                if self.limit is not None and len(data_list) >= self.limit:
                    return data_list
            if self.limit is not None and len(data_list) >= self.limit:
                return data_list
        return data_list

    def _load_data_list(self) -> List[Data]:
        """Parse the raw parquet file and apply ``pre_filter``."""
        parquet_file = pq.ParquetFile(self.raw_paths[0])

        data_list = self._process_parquet(parquet_file)

        return data_list

    def process(self):
        """Processes dataset."""
        data_list = self._load_data_list()

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
