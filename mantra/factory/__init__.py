"""Configuration-driven factory for ready-to-train MANTRA datasets.

A single :class:`DatasetFactoryConfig` selects a dataset variant (unbalanced,
balanced, or one of the subdivided test sets), a model family (a higher-order
``bundle`` or a graph ``representation`` + ``featurization``) and a task;
:func:`build_dataset` then downloads/derives the data, applies every required
pre-transform, and returns a cached PyG dataset.
"""

from mantra.factory.builder import build_dataset, build_pretransform
from mantra.factory.config import DatasetFactoryConfig
from mantra.factory.presets import (
    ALL_VARIANTS,
    BUNDLES,
    FEATURIZATION_SPECS,
    REPRESENTATION_SPECS,
    bundle_required_keys,
)

__all__ = [
    "DatasetFactoryConfig",
    "build_dataset",
    "build_pretransform",
    "bundle_required_keys",
    "ALL_VARIANTS",
    "BUNDLES",
    "REPRESENTATION_SPECS",
    "FEATURIZATION_SPECS",
]
