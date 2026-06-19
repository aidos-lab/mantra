"""Build ready-to-train MANTRA datasets from a declarative config.

The factory composes the representation, featurization, label and
feature-selection transforms in *dependency order* (representation first, so
features that read ``edge_index`` / incidence matrices see them), wires them
as the dataset ``pre_transform``, and returns a cached PyG dataset whose
samples carry everything the target model consumes.
"""

from torch_geometric.transforms import Compose

from mantra.factory.config import DatasetFactoryConfig
from mantra.factory.presets import (
    BUNDLE_SPECS,
    FEATURIZATION_SPECS,
    REPRESENTATION_SPECS,
    TASK_SOURCE,
    build_dataset_for_variant,
)
from mantra.transforms import CreateLabels, SelectFeatures


def _representation_transforms(config: DatasetFactoryConfig):
    """Representation transform(s) for the selected model family."""
    if config.is_bundle:
        return BUNDLE_SPECS[config.bundle]()
    return [REPRESENTATION_SPECS[config.representation]()]


def _featurization_transform(config: DatasetFactoryConfig):
    """Featurization transform appropriate for the model family."""
    spec = FEATURIZATION_SPECS[config.featurization]
    factory = spec.sc if config.is_bundle else spec.graph
    return factory(config.feature_dim)


def build_pretransform(config: DatasetFactoryConfig):
    """Build the pre-transform pipeline and the label transform.

    Returns
    -------
    (Compose, CreateLabels)
        The composed pre-transform and the (referenced) label transform, whose
        ``label_to_index`` is populated while the dataset is processed.
    """
    representation_str = "sc" if config.is_bundle else "graph"
    spec = FEATURIZATION_SPECS[config.featurization]

    label_transform = CreateLabels(source=TASK_SOURCE[config.task])
    select_transform = SelectFeatures(
        src=spec.src,
        dst=None,
        representation=representation_str,
    )

    transforms = [
        *_representation_transforms(config),
        _featurization_transform(config),
        label_transform,
        select_transform,
    ]
    return Compose(transforms), label_transform


def _reconstruct_label_to_index(dataset):
    """Recover the label -> index map from cached samples.

    ``CreateLabels`` only populates its mapping while ``process`` runs, so a
    dataset loaded from cache would otherwise expose an empty map. The stored
    ``label``/``y`` pair makes the mapping recoverable on every load.
    """
    label_to_index = {}
    for data in dataset:
        if "label" in data and "y" in data:
            label_to_index[data.label] = int(data.y.item())
    return label_to_index


def build_dataset(config: DatasetFactoryConfig):
    """Build a training-ready dataset for ``config``.

    The returned dataset's samples carry the representation, features and
    label required by the chosen model family. The dataset also exposes a
    ``label_to_index`` attribute mapping source labels to class indices.
    """
    pre_transform, _ = build_pretransform(config)

    dataset = build_dataset_for_variant(
        config.variant,
        dimension=config.dimension,
        root=config.root,
        version=config.version,
        seed=config.seed,
        name=config.slug(),
        pre_transform=pre_transform,
        transform=None,
        force_reload=config.force_reload,
        local_path=config.local_path,
    )

    dataset.label_to_index = _reconstruct_label_to_index(dataset)
    return dataset
