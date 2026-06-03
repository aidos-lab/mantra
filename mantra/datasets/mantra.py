import os

from torch_geometric.data import (
    download_url,
    extract_gz,
)
import shutil

from mantra.datasets.utils import _get_mantra_dataset_url
from mantra.datasets.base import ManifoldTriangulations

class MANTRA(ManifoldTriangulations):
    """Base class for MANTRA."""

    def __init__(
        self,
        root,
        dimension=2,
        version="latest",
        balanced=False,
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
        dimension : int
            Dimension of manifold triangulations to load. Currently, only
            2 or 3 are supported, denoting 2-manifolds (i.e., surfaces)
            and 3-manifolds, respectively.

        version : str
            Version of the dataset to use. The version should correspond to a
            released version of the dataset, all of which can be found
            `on GitHub <https://github.com/aidos-lab/mantra/releases>`__.
            By default, the latest version will be downloaded. Unless
            specific reproducibility requirements are to be met, using
            `latest` is recommended.

        balanced : bool
            If True, download the balanced variant of the dataset.
            Balanced datasets have been augmented via Pachner moves
            so that all manifold classes have roughly equal
            representation and vertex count distributions.

        local_path : str or None
            If set, use a local JSON file instead of downloading from
            GitHub. The file will be copied into the raw directory.
            Useful for testing locally generated datasets.

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
        """
        assert dimension in [2, 3]

        self.dimension = dimension
        self.balanced = balanced
        self.version = version

        self.local_path = os.path.abspath(local_path) if local_path else None
        self.url = _get_mantra_dataset_url(version, dimension, balanced)

        super().__init__(root, version, name,
                        local_path, transform,
                        pre_transform, pre_filter, force_reload)


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
        suffix = "_balanced" if self.balanced else ""
        return [f"{self.dimension}_manifolds{suffix}.json"]

    def download(self):
        """Download dataset depending on specified version."""
        if self.local_path is not None:
            dst = os.path.join(self.raw_dir, self.raw_file_names[0])
            shutil.copy2(self.local_path, dst)
        else:
            path = download_url(self.url, self.raw_dir)
            extract_gz(path, self.raw_dir)
            os.unlink(path)
    @property
    def processed_file_names(self):
        """Return process file names.

        Stores the processed data in a file. If this file is present in the
        `processed` folder, processing will typically be skipped.
        """
        suffix = "_balanced" if self.balanced else ""
        return [f"data_{self.dimension}{suffix}.pt"]