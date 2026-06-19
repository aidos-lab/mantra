"""Subdivided MANTRA test datasets.

This module derives *subdivided* manifold-triangulation datasets on the fly
from the base (unbalanced) MANTRA data. Rather than hosting separate
artifacts, the subdivided variants are reproduced deterministically from the
downloaded base JSON using the subdivision pipeline in
:mod:`mantra.generate_subdivided_datasets`, then cached like any other
processed dataset.

The variants are intended as held-out *test* datasets that probe how well a
model trained on the base triangulations generalises to finer triangulations
of the same manifolds. The supported variants are ``barycentric``,
``stellar_full``, ``stellar_0.75`` and ``graded``; see
:data:`mantra.factory.presets.SUBDIVISION_PRESETS` for the canonical
generation parameters.
"""

import json
import os
import shutil
import tempfile

from torch_geometric.data import (
    download_url,
    extract_gz,
)

from mantra.datasets.base import ManifoldTriangulations
from mantra.datasets.utils import _get_mantra_dataset_url
from mantra.generate_subdivided_datasets import generate_levels


class MANTRASubdivided(ManifoldTriangulations):
    """Subdivided manifold triangulations derived from the base MANTRA data.

    On first use the base (unbalanced) MANTRA JSON for the requested
    dimension is downloaded (or copied from ``local_path``), every entry is
    subdivided according to ``subdivision_kwargs`` and the result is cached.
    Subsequent instantiations load the cached processed file.
    """

    def __init__(
        self,
        root,
        dimension=2,
        variant="barycentric",
        subdivision_kwargs=None,
        version="latest",
        name=None,
        local_path=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        """Create a subdivided MANTRA dataset.

        Parameters
        ----------
        dimension : int
            Dimension of the manifold triangulations (2 or 3).

        variant : str
            Name of the subdivision variant. Used purely for naming the raw
            and processed files so that variants coexist on disk; the actual
            geometry is controlled by ``subdivision_kwargs``.

        subdivision_kwargs : dict or None
            Keyword arguments forwarded to
            :func:`mantra.generate_subdivided_datasets.generate_levels`
            (e.g. ``mode``, ``n_levels``, ``seed``, ``min_vertices``,
            ``min_class_count``, ``n_smallest``). Must produce exactly one
            output level. If ``None``, a single full barycentric pass is
            applied.

        version, name, local_path, transform, pre_transform, pre_filter,
        force_reload :
            As for :class:`mantra.datasets.MANTRA`. ``local_path``, if set,
            must point to a *base* (unsubdivided) MANTRA JSON; subdivision is
            still applied on top of it.
        """
        assert dimension in [2, 3]

        self.dimension = dimension
        self.variant = variant
        self.subdivision_kwargs = (
            dict(subdivision_kwargs)
            if subdivision_kwargs is not None
            else {"mode": "full_barycentric", "n_levels": 1}
        )
        self.version = version

        self.local_path = os.path.abspath(local_path) if local_path else None
        # The base (unbalanced) dataset is the source for every subdivision.
        self.url = _get_mantra_dataset_url(version, dimension, balanced=False)

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
            return f"/mantra/{self.dimension}D"
        else:
            return f"/mantra/{self.version}/{self.dimension}D"

    @property
    def raw_file_names(self):
        """Disambiguated raw file name for this subdivision variant.

        Unlike the level-keyed names produced by the generation CLI (where
        full barycentric and full stellar both collapse to ``bary_1``), the
        variant name is encoded directly so the four variants never collide.
        """
        return [f"{self.dimension}_manifolds_{self.variant}.json"]

    @property
    def processed_file_names(self):
        """Return processed file name for this subdivision variant."""
        return [f"data_{self.dimension}_{self.variant}.pt"]

    def _load_base_entries(self, scratch_dir):
        """Fetch the base (unsubdivided) entries into ``scratch_dir``."""
        base_name = f"{self.dimension}_manifolds.json"
        base_path = os.path.join(scratch_dir, base_name)
        if self.local_path is not None:
            shutil.copy2(self.local_path, base_path)
        else:
            gz_path = download_url(self.url, scratch_dir)
            extract_gz(gz_path, scratch_dir)
            os.unlink(gz_path)
        with open(base_path) as f:
            return json.load(f)

    def download(self):
        """Download the base data and write the subdivided raw JSON.

        The base dataset is materialised in a temporary directory, subdivided
        via :func:`generate_levels`, and the single resulting level is written
        to the variant-specific raw file. Only the subdivided JSON is kept in
        ``raw_dir``.
        """
        with tempfile.TemporaryDirectory() as scratch_dir:
            base_entries = self._load_base_entries(scratch_dir)

        outputs = generate_levels(base_entries, **self.subdivision_kwargs)
        if len(outputs) != 1:
            raise ValueError(
                f"Subdivision for variant {self.variant!r} must produce "
                f"exactly one level, got keys {sorted(outputs)}. Adjust "
                f"`subdivision_kwargs` (e.g. set a single n_levels)."
            )
        entries = next(iter(outputs.values()))

        dst = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(dst, "w") as f:
            json.dump(entries, f)
