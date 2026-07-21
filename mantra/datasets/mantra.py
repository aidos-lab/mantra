"""Datasets module
This module contains datasets describing triangulations of manifolds,
following the API of `pytorch-geometric`.
"""

import inspect
import json
import os
import shutil
import warnings

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)
from tqdm import tqdm

from mantra.augmentations.balancing import balance_dataset
from mantra.datasets.utils import (
    _find_cached_version,
    _get_mantra_dataset_url,
    _resolve_latest_version,
)

# Keyword arguments forwardable to balance_dataset, derived from its
# signature so the two cannot drift apart. The seed comes from the
# dataset's own seed parameter and the vertex cap from the dataset's
# top-level max_vertices parameter.
BALANCE_KWARGS_KEYS = set(inspect.signature(balance_dataset).parameters) - {
    "dataset",
    "seed",
    "max_vertices",
}


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
        balance_kwargs=None,
        max_vertices=None,
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
            `latest` is recommended; it is resolved to the actual release
            tag on construction, so new releases are picked up
            automatically.
        dimension : int
            Dimension of manifold triangulations to load. Currently, only
            2 or 3 are supported, denoting 2-manifolds (i.e., surfaces)
            and 3-manifolds, respectively.
        balanced : bool
            If True, balance the dataset during processing via Pachner
            move augmentation (see
            :func:`mantra.augmentations.balancing.balance_dataset`), so
            that all manifold classes have equal representation.
            Isomorphic duplicates created by the augmentation are removed
            with the deduplication machinery in
            ``mantra.utils.deduplication``. The result is cached in the
            processed directory; the first run can take a while,
            especially for the 3D dataset. Note that augmented
            near-duplicates of the same source triangulation may end up
            in different splits when the balanced dataset is split
            downstream.
        balance_kwargs : dict or None
            Additional arguments forwarded to
            :func:`mantra.augmentations.balancing.balance_dataset` when
            ``balanced`` is True. Allowed keys: ``target_count``,
            ``n_moves``, ``use_topology_changes``, ``verbose``. The seed
            is always taken from ``seed`` and the vertex cap from
            ``max_vertices``.
        max_vertices : int or None
            If set, keep only triangulations with at most this many
            vertices. With ``balanced=True`` the cap is enforced inside
            the balancing itself (as a prefilter and during
            augmentation), so the classes stay balanced under the cap.
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
        self.balance_kwargs = dict(balance_kwargs or {})
        if self.balance_kwargs and not balanced:
            raise ValueError(
                "balance_kwargs requires balanced=True; got "
                f"{sorted(self.balance_kwargs)} with balanced=False."
            )
        if "seed" in self.balance_kwargs:
            raise ValueError(
                "Do not pass 'seed' in balance_kwargs; balancing uses the "
                "dataset's seed parameter."
            )
        if "max_vertices" in self.balance_kwargs:
            raise ValueError(
                "Do not pass 'max_vertices' in balance_kwargs; use the "
                "top-level max_vertices parameter instead."
            )
        unknown_keys = set(self.balance_kwargs) - BALANCE_KWARGS_KEYS
        if unknown_keys:
            raise ValueError(
                f"Unknown balance_kwargs keys {sorted(unknown_keys)}; "
                f"allowed keys are {sorted(BALANCE_KWARGS_KEYS)}."
            )

        self.max_vertices = max_vertices
        self.local_path = os.path.abspath(local_path) if local_path else None
        resolved_from_latest = False
        if version == "latest" and self.local_path is None:
            version = _resolve_latest_version()
            resolved_from_latest = version != "latest"
            if not resolved_from_latest:
                # Offline: fall back to the newest locally cached release
                # so a warm cache keeps working without network access.
                cached = _find_cached_version(root, dimension)
                if cached is not None:
                    version = cached
                    resolved_from_latest = True
                    warnings.warn(
                        f"Using locally cached MANTRA release {cached}."
                    )
        self.version = version
        self.seed = seed
        self.balanced = balanced
        self.name = name
        self.dimension = dimension
        # Tags resolved from the `latest` alias come from GitHub itself
        # and need no validation round-trip.
        self.url = _get_mantra_dataset_url(
            version, dimension, validate=not resolved_from_latest
        )

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

    def _balance_dir_suffix(self):
        """Suffix encoding the explicitly set data-changing parameters.

        Ensures datasets prepared with different balancing parameters
        or vertex caps are cached in different directories. Every key
        is encoded generically so a newly added parameter can never
        silently share a cache directory; ``verbose`` is excluded since
        it does not change the data.
        """
        params = {
            key: value
            for key, value in self.balance_kwargs.items()
            if key != "verbose"
        }
        if self.max_vertices is not None:
            params["max_vertices"] = self.max_vertices
        parts = [f"{key}{value}" for key, value in sorted(params.items())]
        return "_" + "_".join(parts) if parts else ""

    @property
    def processed_dir(self):
        """Return directory for storing processed data."""
        base_path = os.path.join(self.root, "processed")
        balanced_suffix = "balanced" if self.balanced else "unbalanced"
        balanced_suffix = (
            f"{balanced_suffix}_{self.seed}{self._balance_dir_suffix()}"
        )

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

    def _load_raw_entries(self):
        """Load raw JSON entries, applying the vertex cap and balancing."""
        with open(self.raw_paths[0]) as f:
            inputs = json.load(f)

        if self.balanced:
            # balance_dataset enforces the vertex cap itself, both as a
            # prefilter and during augmentation.
            inputs = balance_dataset(
                inputs,
                seed=self.seed,
                max_vertices=self.max_vertices,
                **self.balance_kwargs,
            )
        elif self.max_vertices is not None:
            inputs = [
                e for e in inputs if e["n_vertices"] <= self.max_vertices
            ]

        return inputs

    def process(self):
        """Processes dataset."""
        inputs = self._load_raw_entries()

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
