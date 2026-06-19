"""Configuration object for the dataset factory."""

from dataclasses import dataclass
from typing import Optional, Union

from mantra.factory.presets import (
    ALL_VARIANTS,
    BUNDLES,
    FEATURIZATION_SPECS,
    REPRESENTATION_SPECS,
    TASK_SOURCE,
)
from mantra.tasks.task_types import TaskType


def _coerce_task(task: Union[str, TaskType]) -> TaskType:
    if isinstance(task, TaskType):
        return task
    try:
        return TaskType(task)
    except ValueError as exc:
        valid = sorted(t.value for t in TASK_SOURCE)
        raise ValueError(
            f"Unknown task {task!r}; choose one of {valid}."
        ) from exc


@dataclass
class DatasetFactoryConfig:
    """Declarative description of a ready-to-train MANTRA dataset.

    Exactly one model family must be selected: either a higher-order
    ``bundle`` (``"san"``/``"sccnn"``/``"cwn"``) **or** a graph
    ``representation`` paired with a ``featurization``.

    Parameters
    ----------
    dimension : int
        2 or 3.
    variant : str
        Dataset variant; one of :data:`mantra.factory.presets.ALL_VARIANTS`.
    task : str or TaskType
        Classification target (``name``, ``orientability``, ``betti_numbers``).
    bundle : str or None
        Higher-order model bundle. Mutually exclusive with
        ``representation``/``featurization``.
    representation : str or None
        Graph representation (``one_skeleton``/``dual_graph``/``hasse_diagram``).
    featurization : str or None
        Feature scheme for the graph representation.
    root : str
        Dataset root directory.
    version : str
        Dataset release version (or ``"latest"``).
    feature_dim : int
        Width of random features (ignored by featurizations with a fixed width).
    seed : int
        Seed for deterministic subdivision generation.
    local_path : str or None
        Local base JSON to use instead of downloading (for testing).
    force_reload : bool
        Re-run processing even if a cached file exists.
    """

    dimension: int
    variant: str
    task: Union[str, TaskType] = TaskType.NAME
    bundle: Optional[str] = None
    representation: Optional[str] = None
    featurization: Optional[str] = None
    root: str = "./data"
    version: str = "latest"
    feature_dim: int = 8
    seed: int = 42
    local_path: Optional[str] = None
    force_reload: bool = False

    def __post_init__(self):
        if self.dimension not in (2, 3):
            raise ValueError(
                f"dimension must be 2 or 3, got {self.dimension}."
            )

        if self.variant not in ALL_VARIANTS:
            raise ValueError(
                f"Unknown variant {self.variant!r}; choose one of "
                f"{sorted(ALL_VARIANTS)}."
            )

        self.task = _coerce_task(self.task)

        # ``featurization`` is shared by both families (and may be overridden
        # for a bundle), so the family is decided by ``bundle`` vs.
        # ``representation`` only.
        is_bundle = self.bundle is not None
        is_graph = self.representation is not None
        if is_bundle and is_graph:
            raise ValueError(
                "Specify either `bundle` (higher-order) or "
                "`representation` (graph), not both."
            )
        if not is_bundle and not is_graph:
            raise ValueError(
                "Specify a model family: either `bundle` or "
                "`representation`."
            )

        if is_bundle:
            if self.bundle not in BUNDLES:
                raise ValueError(
                    f"Unknown bundle {self.bundle!r}; choose one of "
                    f"{sorted(BUNDLES)}."
                )
            # Default the bundle featurization to random per-simplex features.
            if self.featurization is None:
                self.featurization = "random"
            self._require_sc_featurization()
        else:
            if self.representation not in REPRESENTATION_SPECS:
                raise ValueError(
                    f"Unknown representation {self.representation!r}; choose "
                    f"one of {sorted(REPRESENTATION_SPECS)}."
                )
            if self.featurization is None:
                self.featurization = "random"
            self._require_graph_featurization()

    @property
    def is_bundle(self) -> bool:
        """Whether this config targets a higher-order model bundle."""
        return self.bundle is not None

    def _featurization_spec(self):
        if self.featurization not in FEATURIZATION_SPECS:
            raise ValueError(
                f"Unknown featurization {self.featurization!r}; choose one of "
                f"{sorted(FEATURIZATION_SPECS)}."
            )
        return FEATURIZATION_SPECS[self.featurization]

    def _require_graph_featurization(self):
        if self._featurization_spec().graph is None:
            raise ValueError(
                f"Featurization {self.featurization!r} is not available for "
                f"graph representations."
            )

    def _require_sc_featurization(self):
        if self._featurization_spec().sc is None:
            raise ValueError(
                f"Featurization {self.featurization!r} is not available for "
                f"higher-order (simplicial-complex) bundles."
            )

    def slug(self) -> str:
        """Deterministic on-disk name so distinct configs coexist."""
        family = self.bundle if self.is_bundle else self.representation
        return (
            f"{self.variant}-{family}-{self.featurization}-{self.task.value}"
        )
