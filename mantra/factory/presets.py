"""Registries that map factory-config choices to concrete dataset behaviour.

These tables are the single source of truth for the dataset *variants*, graph
*representations*, *featurizations*, and higher-order model *bundles* that the
:func:`mantra.factory.build_dataset` factory understands. Keeping them here
(rather than inline in the builder) makes the supported configuration surface
explicit and easy to extend or test.
"""

from mantra.datasets import MANTRA
from mantra.datasets.subdivided import MANTRASubdivided
from mantra.representations import (
    AdjacencySimplicialComplex,
    DualGraph,
    HasseDiagram,
    IncidenceSimplicialComplex,
    OneSkeleton,
)
from mantra.representations.simplicial_connectivity import (
    DownLaplacianSimplicialComplex,
    HodgeLaplacianSimplicialComplex,
    UpLaplacianSimplicialComplex,
)
from mantra.tasks.task_types import TaskType
from mantra.transforms import (
    EffectiveResistanceEmbedding,
    MomentCurveEmbedding,
    NodeDegreeTransform,
    NodeRandomTransform,
)

# --------------------------------------------------------------------------- #
# Dataset variants
# --------------------------------------------------------------------------- #

#: Variant names served directly by the base :class:`MANTRA` dataset.
_BASE_VARIANTS = {
    "unbalanced": {"balanced": False},
    "balanced": {"balanced": True},
}

#: Subdivided test-dataset variants. Each maps to the keyword arguments for
#: :func:`mantra.generate_subdivided_datasets.generate_levels` (excluding the
#: shared ``seed``, which is supplied from the config). Per-dimension overrides
#: bound the cost of the (much larger) 3D base data via smallest-cohort
#: generation, while 2D subdivides every entry.
#: ``graded`` always uses the smallest-cohort generator (``n_smallest``): it is
#: the only code path in ``generate_levels`` that emits a graded level, and it
#: keeps the grown-to-target test set to a bounded, per-class size.
SUBDIVISION_PRESETS = {
    "barycentric": {"mode": "full_barycentric", "n_levels": 1},
    "stellar_full": {"mode": "stellar", "n_levels": 1.0},
    "stellar_0.75": {"mode": "stellar", "n_levels": 0.75},
    "graded": {"mode": "graded", "n_levels": 16, "n_smallest": 50},
}

#: Per-(variant, dimension) overrides merged onto :data:`SUBDIVISION_PRESETS`.
_SUBDIVISION_DIM_OVERRIDES = {
    # 3D barycentric explodes (24 cells / tetrahedron); cap per-class size.
    3: {
        "barycentric": {"n_smallest": 50},
        "stellar_full": {"n_smallest": 50},
        "stellar_0.75": {"n_smallest": 50},
        "graded": {"n_levels": 32, "n_smallest": 50},
    },
}

SUBDIVIDED_VARIANTS = tuple(SUBDIVISION_PRESETS)
BASE_VARIANTS = tuple(_BASE_VARIANTS)
ALL_VARIANTS = BASE_VARIANTS + SUBDIVIDED_VARIANTS


def subdivision_kwargs(variant, dimension, seed):
    """Resolve the ``generate_levels`` kwargs for a subdivided variant."""
    kwargs = dict(SUBDIVISION_PRESETS[variant])
    kwargs.update(
        _SUBDIVISION_DIM_OVERRIDES.get(dimension, {}).get(variant, {})
    )
    kwargs["seed"] = seed
    return kwargs


def build_dataset_for_variant(
    variant,
    *,
    dimension,
    root,
    version,
    seed,
    name,
    pre_transform,
    transform,
    force_reload,
    local_path=None,
):
    """Instantiate the correct dataset class for a variant."""
    if variant in _BASE_VARIANTS:
        return MANTRA(
            root=root,
            dimension=dimension,
            version=version,
            name=name,
            local_path=local_path,
            pre_transform=pre_transform,
            transform=transform,
            force_reload=force_reload,
            **_BASE_VARIANTS[variant],
        )
    if variant in SUBDIVISION_PRESETS:
        return MANTRASubdivided(
            root=root,
            dimension=dimension,
            variant=variant,
            subdivision_kwargs=subdivision_kwargs(variant, dimension, seed),
            version=version,
            name=name,
            local_path=local_path,
            pre_transform=pre_transform,
            transform=transform,
            force_reload=force_reload,
        )
    raise ValueError(
        f"Unknown variant {variant!r}; choose one of {sorted(ALL_VARIANTS)}."
    )


# --------------------------------------------------------------------------- #
# Graph representations
# --------------------------------------------------------------------------- #

#: Graph-representation transforms (produce ``edge_index``).
REPRESENTATION_SPECS = {
    "one_skeleton": OneSkeleton,
    "dual_graph": DualGraph,
    "hasse_diagram": HasseDiagram,
}


# --------------------------------------------------------------------------- #
# Featurizations
# --------------------------------------------------------------------------- #


class _Featurization:
    """A featurization choice and the attribute it writes.

    ``graph`` / ``sc`` are factories returning the transform for the graph and
    simplicial-complex representations respectively (``None`` if unsupported).
    ``src`` is the ``Data`` attribute the transform writes, consumed by
    :class:`~mantra.transforms.SelectFeatures`.
    """

    def __init__(self, src, graph, sc):
        self.src = src
        self.graph = graph
        self.sc = sc


#: ``feature_dim`` is threaded in for the random featurization.
FEATURIZATION_SPECS = {
    "random": _Featurization(
        src="random_features",
        graph=lambda dim: NodeRandomTransform(dim=dim, propagate=False),
        sc=lambda dim: NodeRandomTransform(dim=dim, propagate=True),
    ),
    "degree": _Featurization(
        src="degree",
        graph=lambda dim: NodeDegreeTransform(),
        sc=None,
    ),
    "moment_curve": _Featurization(
        src="moment_curve_embedding",
        graph=lambda dim: MomentCurveEmbedding(propagate=False),
        sc=lambda dim: MomentCurveEmbedding(propagate=True),
    ),
    "effective_resistance": _Featurization(
        src="er",
        graph=None,
        sc=lambda dim: EffectiveResistanceEmbedding(),
    ),
}


# --------------------------------------------------------------------------- #
# Higher-order model bundles
# --------------------------------------------------------------------------- #

# Unsigned connectivity matches the published benchmark configs. Every bundle
# includes the incidence matrices so that propagated per-simplex features can
# be derived; models simply ignore connectivity keys they do not read.
_SIGNED = False


def _bundle_transforms(extra):
    """Bundle = incidence (for feature propagation) + model-specific matrices."""
    return [IncidenceSimplicialComplex(signed=_SIGNED)] + extra


#: Ordered connectivity transforms per higher-order model bundle.
BUNDLE_SPECS = {
    # SAN reads edge (1-cell) features and the up/down Laplacians on edges.
    "san": lambda: _bundle_transforms(
        [
            UpLaplacianSimplicialComplex(signed=_SIGNED),
            DownLaplacianSimplicialComplex(signed=_SIGNED),
        ]
    ),
    # SCCNN reads features on all dimensions plus Hodge/up/down Laplacians and
    # both incidence (boundary) matrices.
    "sccnn": lambda: _bundle_transforms(
        [
            HodgeLaplacianSimplicialComplex(signed=_SIGNED),
            UpLaplacianSimplicialComplex(signed=_SIGNED),
            DownLaplacianSimplicialComplex(signed=_SIGNED),
        ]
    ),
    # CWN reads features on all dimensions plus the upper adjacency on edges
    # and both incidence matrices.
    "cwn": lambda: _bundle_transforms(
        [AdjacencySimplicialComplex(signed=_SIGNED)]
    ),
}

BUNDLES = tuple(BUNDLE_SPECS)


def bundle_required_keys(bundle, dimension):
    """Minimal set of ``Data`` keys each model's ``forward`` reads.

    This is the verifiable contract exercised by the tests: a dataset built
    for ``bundle`` must contain (at least) these keys. ``up_laplacian_2`` is
    only present/needed for 3D (``sc_order > 2``).
    """
    if bundle == "san":
        return {"x_1", "up_laplacian_1", "down_laplacian_1"}
    if bundle == "sccnn":
        keys = {
            "x_0",
            "x_1",
            "x_2",
            "hodge_laplacian_0",
            "down_laplacian_1",
            "up_laplacian_1",
            "down_laplacian_2",
            "incidence_1",
            "incidence_2",
        }
        if dimension > 2:
            keys.add("up_laplacian_2")
        return keys
    if bundle == "cwn":
        return {
            "x_0",
            "x_1",
            "x_2",
            "adjacency_1",
            "incidence_1",
            "incidence_2",
        }
    raise ValueError(
        f"Unknown bundle {bundle!r}; choose one of {sorted(BUNDLES)}."
    )


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

#: Map a task to the ``Data`` attribute used as the classification source.
TASK_SOURCE = {
    TaskType.NAME: "name",
    TaskType.ORIENTABILITY: "orientable",
    TaskType.BETTI_NUMBERS: "betti_numbers",
}
