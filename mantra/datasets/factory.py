"""High-level factory for assembling pre-transformed MANTRA datasets.

``make_dataset(...)`` composes a deterministic ``pre_transform`` pipeline from
the cross-product of (representation, featurization, task) and returns a
ready-to-use :class:`ManifoldTriangulations`. Each combination is cached in its
own ``processed/`` subdirectory so variants coexist on disk.

Labels are assigned in **canonical** space (the indices defined by
``NAME_TO_CLASS_2M`` / ``NAME_TO_CLASS_3M``); callers (e.g. the
mantra-pp-benchmarks training scripts) may filter the dataset and then remap
to a compact ``0..N-1`` space for the model output.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from torch_geometric.transforms import AddRandomWalkPE, BaseTransform, Compose

from mantra.datasets.base import ManifoldTriangulations
from mantra.representations.dual_graph import DualGraph
from mantra.representations.hasse_diagram import HasseDiagram
from mantra.representations.one_skeleton import OneSkeleton
from mantra.representations.simplicial_connectivity import (
    AdjacencySimplicialComplex,
    DownLaplacianSimplicialComplex,
    HodgeLaplacianSimplicialComplex,
    IncidenceSimplicialComplex,
    UpLaplacianSimplicialComplex,
)
from mantra.transforms.attribute_transform import (
    NodeDegreeTransform,
    NodeRandomTransform,
)
from mantra.transforms.moment_curve_embedding import MomentCurveEmbedding
from mantra.transforms.select_features import SelectFeatures
from mantra.transforms.structural_transforms import SetNumNodesTransform
from mantra.transforms.task_transforms import (
    BettiToClassTransform,
    NameToClass2MTransform,
    NameToClass3MTransform,
    OrientableToClassTransform,
)


Representation = Literal["one_skeleton", "dual", "hasse"]
Featurization = Literal["random", "node_degree", "mc", "rw"]
Task = Literal["name", "orientability", "betti"]
BundleName = Literal["cwn", "sccnn", "san"]


@dataclass(frozen=True)
class ModelBundle:
    """A higher-order model's pre_transform recipe.

    A bundle replaces the single ``representation`` choice with an *ordered*
    stack of simplicial-complex connectivity transforms plus the rank-aware
    featurization those models consume. ``sc_representations`` MUST start with
    ``"incidence_sc"`` because the propagated featurizations key their per-rank
    features off the incidence matrices it writes.
    """

    name: str
    sc_representations: tuple[str, ...]
    signed: bool
    propagate: bool
    select_representation: str  # "sc"


# Maps the rank-aware higher-order models to the exact ordered connectivity
# stack each one expects (verified against the benchmarks precompute scripts).
MODEL_BUNDLES: dict[str, ModelBundle] = {
    "cwn": ModelBundle(
        "cwn",
        sc_representations=("incidence_sc", "adjacency_sc"),
        signed=True,
        propagate=True,
        select_representation="sc",
    ),
    "sccnn": ModelBundle(
        "sccnn",
        sc_representations=(
            "incidence_sc",
            "up_laplacian_sc",
            "down_laplacian_sc",
            "hodge_laplacian_sc",
        ),
        signed=True,
        propagate=True,
        select_representation="sc",
    ),
    "san": ModelBundle(
        "san",
        sc_representations=(
            "incidence_sc",
            "up_laplacian_sc",
            "down_laplacian_sc",
        ),
        signed=True,
        propagate=True,
        select_representation="sc",
    ),
}


# Constructors for each simplicial-complex connectivity key. ``signed`` is
# threaded from the bundle; ``AdjacencySimplicialComplex`` has no ``index`` arg.
_SC_REP_BUILDERS: dict[str, Callable[[bool], BaseTransform]] = {
    "incidence_sc": lambda signed: IncidenceSimplicialComplex(signed=signed),
    "adjacency_sc": lambda signed: AdjacencySimplicialComplex(signed=signed),
    "up_laplacian_sc": lambda signed: UpLaplacianSimplicialComplex(signed=signed),
    "down_laplacian_sc": lambda signed: DownLaplacianSimplicialComplex(signed=signed),
    "hodge_laplacian_sc": lambda signed: HodgeLaplacianSimplicialComplex(signed=signed),
}

# Only the two dict-producing featurizations support ``propagate=True`` (they
# emit per-rank features that ``SelectFeatures(representation='sc')`` writes to
# ``x_0..x_d``). ``node_degree``/``rw`` read ``edge_index`` and are graph-only.
_BUNDLE_FEATURIZATIONS = {"random", "mc"}


_LEGAL_PAIRS: dict[Representation, set[Featurization]] = {
    # The three graph representations produce an ``edge_index``, so all four
    # featurizations are well-defined on top of them.
    "one_skeleton": {"random", "node_degree", "mc", "rw"},
    "dual": {"random", "node_degree", "mc", "rw"},
    "hasse": {"random", "node_degree", "mc", "rw"},
}


def _validate_pair(representation: Representation, featurization: Featurization) -> None:
    if representation not in _LEGAL_PAIRS:
        raise ValueError(
            f"Unknown representation {representation!r}. "
            f"Expected one of {sorted(_LEGAL_PAIRS)}."
        )
    legal = _LEGAL_PAIRS[representation]
    if featurization not in legal:
        raise ValueError(
            f"Combination representation={representation!r}, "
            f"featurization={featurization!r} is not supported. "
            f"Legal featurizations for {representation!r}: {sorted(legal)}."
        )


def _build_representation(representation: Representation) -> BaseTransform:
    if representation == "one_skeleton":
        return OneSkeleton()
    if representation == "dual":
        return DualGraph()
    if representation == "hasse":
        return HasseDiagram()
    raise ValueError(f"Unknown representation: {representation}")


def _build_featurization(
    representation: Representation,
    featurization: Featurization,
    feature_dim: int,
) -> tuple[BaseTransform, BaseTransform]:
    """Return ``(feature_transform, select_features_transform)``."""
    # The graph representations all write features to ``data.x`` (PyG graph
    # convention); the simplicial-complex path is handled separately by the
    # model bundles.
    sf_representation: Literal["graph", "sc"] = "graph"

    if featurization == "random":
        return NodeRandomTransform(dim=feature_dim, propagate=False), SelectFeatures(
            src="random_features", dst=None, representation=sf_representation
        )
    if featurization == "node_degree":
        return NodeDegreeTransform(), SelectFeatures(
            src="degree", dst=None, representation=sf_representation
        )
    if featurization == "mc":
        return MomentCurveEmbedding(propagate=False), SelectFeatures(
            src="moment_curve_embedding", dst=None, representation=sf_representation
        )
    if featurization == "rw":
        # AddRandomWalkPE writes to ``data[attr_name]`` (default
        # ``random_walk_pe``) and reads ``data.edge_index``; its output
        # dimension equals ``walk_length``.
        return AddRandomWalkPE(
            walk_length=feature_dim, attr_name="random_walk_pe"
        ), SelectFeatures(
            src="random_walk_pe", dst=None, representation=sf_representation
        )
    raise ValueError(f"Unknown featurization: {featurization}")


def _build_sc_featurization(
    featurization: Featurization,
    feature_dim: int,
    propagate: bool,
) -> tuple[BaseTransform, BaseTransform]:
    """Return ``(feature_transform, select_features_transform)`` for a bundle.

    Both branches emit a rank-keyed feature dict (when ``propagate`` is True)
    that :class:`SelectFeatures` with ``representation="sc"`` lands on
    ``x_0..x_d``.
    """
    if featurization == "random":
        return NodeRandomTransform(dim=feature_dim, propagate=propagate), SelectFeatures(
            src="random_features", dst=None, representation="sc"
        )
    if featurization == "mc":
        return MomentCurveEmbedding(propagate=propagate), SelectFeatures(
            src="moment_curve_embedding", dst=None, representation="sc"
        )
    raise ValueError(
        f"Featurization {featurization!r} is not supported for model bundles "
        f"(expected one of {sorted(_BUNDLE_FEATURIZATIONS)}). "
        f"node_degree/rw read edge_index and are only valid on the graph "
        f"`representation` path."
    )


def _build_bundle_pre_transform(
    bundle: ModelBundle,
    featurization: Featurization,
    task: Task,
    dimension: int,
    feature_dim: int,
) -> Compose:
    # Ordered connectivity stack first (incidence_sc leads, so the propagated
    # featurization finds the incidence matrices it keys off), then the
    # rank-aware featurization, then SelectFeatures(sc), then the task label.
    # Note: unlike the graph path there is NO SetNumNodesTransform — the SC
    # connectivity transforms read ``triangulation``/``dimension`` directly and
    # never touch ``num_nodes`` (this matches the precompute scripts).
    steps: list[BaseTransform] = [
        _SC_REP_BUILDERS[rep_key](bundle.signed)
        for rep_key in bundle.sc_representations
    ]
    feat, select = _build_sc_featurization(featurization, feature_dim, bundle.propagate)
    steps.append(feat)
    steps.append(select)
    steps.append(_build_task_transform(task, dimension))
    return Compose(steps)


def _build_task_transform(task: Task, dimension: int) -> BaseTransform:
    if task == "name":
        if dimension == 2:
            return NameToClass2MTransform()
        if dimension == 3:
            return NameToClass3MTransform()
        raise ValueError(f"task='name' requires dimension in (2, 3), got {dimension}")
    if task == "orientability":
        return OrientableToClassTransform()
    if task == "betti":
        return BettiToClassTransform(manifold_dim=dimension)
    raise ValueError(
        f"Unknown task {task!r}. Expected one of 'name', 'orientability', 'betti'."
    )


def _derive_cache_name(
    representation: Representation | None,
    featurization: Featurization,
    task: Task,
    feature_dim: int,
    shortest_path_distance: bool,
    subdivision_level: int | None,
    bundle: ModelBundle | None = None,
) -> str:
    feat_tag = (
        f"{featurization}{feature_dim}" if featurization != "node_degree" else "node_degree"
    )
    if bundle is not None:
        # Encode the bundle identity (name + signed + propagate) so SC caches
        # stay disjoint from the graph-representation caches on disk.
        parts = [
            f"bundle_{bundle.name}",
            feat_tag,
            f"signed{int(bundle.signed)}",
            f"prop{int(bundle.propagate)}",
            f"task_{task}",
        ]
    else:
        parts = [representation, feat_tag, f"task_{task}"]
    if shortest_path_distance:
        parts.append("spd")
    if subdivision_level is not None:
        parts.append(f"bary{subdivision_level}")
    return "__".join(parts)


def _build_pre_transform(
    representation: Representation,
    featurization: Featurization,
    task: Task,
    dimension: int,
    feature_dim: int,
    shortest_path_distance: bool,
    spd_transform: Optional[BaseTransform],
) -> Compose:
    # Representation must run before SetNumNodesTransform because OneSkeleton's
    # from_networkx step writes its own `num_nodes` and asserts the key isn't
    # already present. Running SetNumNodesTransform afterwards is consistent
    # for all four representations: for OneSkeleton it re-writes the
    # already-correct value; for Dual/Hasse it pulls the simplex-count
    # ``n_vertices`` those representations have just overwritten; for
    # triangulation it's the only thing that sets num_nodes.
    steps: list[BaseTransform] = [_build_representation(representation)]
    steps.append(SetNumNodesTransform())
    feat, select = _build_featurization(representation, featurization, feature_dim)
    steps.append(feat)
    steps.append(select)
    if shortest_path_distance:
        if spd_transform is None:
            raise ValueError(
                "shortest_path_distance=True but no spd_transform was provided. "
                "Pass an instantiated GraphormerSPDPreTransform via spd_transform=..."
            )
        steps.append(spd_transform)
    steps.append(_build_task_transform(task, dimension))
    return Compose(steps)


def make_dataset(
    *,
    root: str,
    dimension: int,
    featurization: Featurization,
    representation: Representation | None = None,
    bundle: str | None = None,
    task: Task = "name",
    balanced: bool = True,
    version: str = "latest",
    local_path: Optional[str] = None,
    feature_dim: int = 8,
    shortest_path_distance: bool = False,
    spd_transform: Optional[BaseTransform] = None,
    pre_filter: Optional[Callable] = None,
    force_reload: bool = False,
    name: Optional[str] = None,
    subdivision_level: Optional[int] = None,
) -> ManifoldTriangulations:
    """Build a pre-transformed :class:`ManifoldTriangulations`.

    Exactly one of ``representation`` (graph path) or ``bundle`` (higher-order
    simplicial-complex path) must be given.

    Parameters
    ----------
    root
        Root directory for the on-disk dataset cache (PyG convention).
    dimension
        Manifold dimension; 2 or 3.
    representation
        Graph path. ``"one_skeleton"``, ``"dual"``, or ``"hasse"``.
        Mutually exclusive with ``bundle``.
    bundle
        Higher-order model bundle: ``"cwn"``, ``"sccnn"``, or ``"san"`` (see
        :data:`MODEL_BUNDLES`). Builds an ordered signed simplicial-complex
        connectivity stack with propagated per-rank features on ``x_0..x_d``,
        ready to train that model. Mutually exclusive with ``representation``.
        Only ``featurization`` in ``{"random", "mc"}`` is supported here.
    featurization
        ``"random"``, ``"node_degree"``, ``"mc"``, or ``"rw"`` (random-walk
        positional encoding via :class:`torch_geometric.transforms.AddRandomWalkPE`).
        Bundles accept only ``"random"``/``"mc"``.
    task
        Classification target. ``"name"`` maps to the canonical
        homeomorphism-type indices; ``"orientability"`` to a binary target;
        ``"betti"`` to a Betti-numbers regression target.
    balanced
        Use the balanced dataset variant. Mirrors
        :class:`ManifoldTriangulations` semantics.
    version, local_path
        Passed through to :class:`ManifoldTriangulations`.
    feature_dim
        Output dimension for ``"random"`` and ``"mc"`` features.
    shortest_path_distance
        If True, append the user-provided ``spd_transform`` (e.g.
        ``GraphormerSPDPreTransform`` from mantra-pp-benchmarks) after the
        featurization step.
    spd_transform
        Required if ``shortest_path_distance=True``.
    pre_filter
        Optional PyG pre-filter callable.
    force_reload
        Bypass the on-disk cache.
    name
        Override the auto-derived cache subdirectory name. Useful when a
        caller wants to manually distinguish variants.
    subdivision_level
        If set to a positive integer, pulls (or expects locally) the raw
        file ``{dim}_manifolds{_balanced}_bary{L}.json`` corresponding to
        ``L`` barycentric subdivisions. ``None`` means the original
        triangulation set. Threaded into ``ManifoldTriangulations`` and
        into the auto-derived cache name so subdivision levels coexist.

    Returns
    -------
    ManifoldTriangulations
        A dataset whose ``data.y`` is in **canonical** class space
        (not yet remapped to ``0..N-1`` — that's the caller's job
        after any filtering).
    """
    if (representation is None) == (bundle is None):
        raise ValueError(
            "Pass exactly one of `representation` (graph path) or `bundle` "
            "(simplicial-complex path)."
        )

    if bundle is not None:
        if bundle not in MODEL_BUNDLES:
            raise ValueError(
                f"Unknown bundle {bundle!r}. Expected one of "
                f"{sorted(MODEL_BUNDLES)}. (Graph models such as 'gcn' use the "
                f"`representation` argument, not `bundle`.)"
            )
        model_bundle = MODEL_BUNDLES[bundle]
        if shortest_path_distance:
            raise ValueError(
                "shortest_path_distance is only supported on the graph "
                "`representation` path, not for model bundles."
            )
        pre_transform = _build_bundle_pre_transform(
            bundle=model_bundle,
            featurization=featurization,
            task=task,
            dimension=dimension,
            feature_dim=feature_dim,
        )
    else:
        model_bundle = None
        _validate_pair(representation, featurization)
        pre_transform = _build_pre_transform(
            representation=representation,
            featurization=featurization,
            task=task,
            dimension=dimension,
            feature_dim=feature_dim,
            shortest_path_distance=shortest_path_distance,
            spd_transform=spd_transform,
        )
    auto_name = name or _derive_cache_name(
        representation,
        featurization,
        task,
        feature_dim,
        shortest_path_distance,
        subdivision_level,
        bundle=model_bundle,
    )
    return ManifoldTriangulations(
        root=root,
        dimension=dimension,
        version=version,
        balanced=balanced,
        local_path=local_path,
        pre_transform=pre_transform,
        pre_filter=pre_filter,
        force_reload=force_reload,
        name=auto_name,
        transform=None,
        subdivision_level=subdivision_level,
    )
