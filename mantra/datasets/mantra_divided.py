import os
from enum import Enum
import json
import shutil
from mantra.augmentations.base import Triangulation
from torch.utils import data
from tqdm import tqdm
import random

from torch_geometric.data import (
    download_url,
    extract_gz,
)

from torch_geometric.data import (
    Data,
)

from mantra.datasets.mantra import ManifoldTriangulations
from mantra.datasets.utils import _get_mantra_dataset_url, filter_by_class_count


class SubdivisionType(Enum):
    STELLAR=1
    GRADED=2
    BARYCENTRIC=3

    def __str__(self):
        return self.name.lower()
    
    @staticmethod
    def from_str(sub_name: str):
        for sub in SubdivisionType:
            if str(sub) == sub_name:
                return sub
        raise ValueError(f"There is no Subdivision with name {sub_name}")

class MANTRADivided(ManifoldTriangulations):
    """Dataset of manifold triangulations from the MANTRA benchmark
        with subdivisions on the test set
    """

    def __init__(
        self,
        root,
        division_type: str,
        dimension=2,
        version="latest",
        name=None,
        local_path=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        class_count_filter=None,
        force_reload=False,
        **kwargs
    ):
        """
        Create a new dataset of manifold triangulations.

        Parameters
        ----------
        division_type : str
            Type of division to apply to the triangulations. Options are
            barycentric, graded, stellar.
        dimension : int
            Dimension of manifold triangulations to load. Currently, only
            2 is support denoting 2-manifolds (i.e., surfaces).
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
        class_count_filter : int or None
            If the initial classes should be filtered before constructing the
            subdivisions.
        """
        assert dimension in [2, 3]

        self.dimension = dimension
        self.version = version
        self.class_count_filter = class_count_filter
        self.division_type = SubdivisionType.from_str(division_type)
        self.kwargs = kwargs

        self.local_path = os.path.abspath(local_path) if local_path else None
        self.url = _get_mantra_dataset_url(version, dimension, False)

        super().__init__(
            root,
            version,
            name,
            local_path,
            transform,
            pre_transform,
            pre_filter,
            force_reload,
        )

    def _add_version_to_root(self):
        if self.version == "latest":
            return f"/mantra_divided/{self.dimension}D"
        else:
            return f"/mantra_divided/{self.version}/{self.dimension}D"

    @property
    def raw_file_names(self):
        """Return raw file names.

        Stores the raw file names that need to be present in the raw folder
        for downloading to be skipped. To reference raw file names, use the
        property `self.raw_paths`.
        """
        return [f"{self.dimension}_manifolds.json"]

    @property
    def processed_file_names(self):
        """Return process file names.

        Stores the processed data in a file. If this file is present in the
        `processed` folder, processing will typically be skipped.
        """
        return [f"data_{self.dimension}_{str(self.division_type)}.pt"]

    def download(self):
        """Download dataset depending on specified version."""
        if self.local_path is not None:
            dst = os.path.join(self.raw_dir, self.raw_file_names[0])
            shutil.copy2(self.local_path, dst)
        else:
            path = download_url(self.url, self.raw_dir)
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def _calculate_induced_vertices(self):
        """" This should calculate the number of vertices
            induced by the application of this subdivision

        """

        # TODO: This might change by dimension
        if self.division_type == SubdivisionType.STELLAR:
            return 1
        elif self.division_type == SubdivisionType.GRADED:
            return 1
        else: # self.division_type == SubdivisionType.BARYCENTRIC:
            return 4

    def _subdivide_triangle(self, data, **kwargs):
        triangulation = data["triangulation"]
        rng = random.Random(self.kwargs.get('seed', 42))

        triangle = Triangulation(triangulation, rng)
        fn_str = f"{str(self.division_type)}_subdivision"
        fn = getattr(triangle, fn_str)

        new_triangulation, n_v = fn(kwargs)

        new_entry = triangulation.copy()
        new_entry["triangulation"] = new_triangulation
        new_entry["n_vertices"] = n_v

        return new_entry


    def _apply_subdivision(self, data_list):
        """ Apply the subdivision to the data_list

        """
        if self.division_type == SubdivisionType.BARYCENTRIC:
            rounds = self.kwargs.get("round", 1)
            for _ in range(rounds):
                data_list = [self.subdivide_triangle(tri) for tri in data_list]
        elif self.division_type == SubdivisionType.STELLAR:
            fraction = self.kwargs.get("fraction", 1)
            data_list = [self.subdivive_triangle(tri, fraction=fraction) for tri in data_list]
        else: # This is the graded
            min_vertices = self.kwargs.get("min_vertices", 1)
            data_list = [self.subdivide_triangle(tri, over_vrtx_cnt=min_vertices) for tri in data_list]
        
        return data_list



    def process(self):
        """Processes dataset."""
        with open(self.raw_paths[0]) as f:
            inputs = json.load(f)

        data_list = [Data(**el) for el in inputs]

        # Filter by homeomorphism type 
        data_list, _ =  filter_by_class_count(data_list, "name", self.class_count_filter)
        
        # Apply the select subdivison algorithm
        data_list = self._apply_subdivision(data_list)

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
