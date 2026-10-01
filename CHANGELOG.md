# CHANGELOG

This is a [changelog](https://keepachangelog.com/) of all notable
changes to this project. We adhere to [Semantic Versioning](https://semver.org/).

# Unreleased

## Added

- `PachnerWalkDataset(size_targets=[n, ...])`, size-targeted training
  walks (2D only): every entry of the splits in `size_splits` (default
  train and val) is followed by one refining walk per target, stopped
  exactly at `n` vertices as the `target` of the `pachner` OOD split
  is, optionally mixed by `size_mix` 2-2 flips per vertex; with
  `keep_base=False` the source entries are dropped, leaving a split
  in which every class appears at every size equally often. Entries carry
  `walk_base`/`walk_step` like step walks, so same-walk pairs link a
  source to its larger copies. `pachner_walk_to` is the shared walk.

- `MantraDataset(division_type="pachner")` (2D only), an OOD split built
  by a random Pachner walk instead of a subdivision: `target` or
  `match={"division_type": ..., ...}` fixes the vertex count the walk
  stops at (`match` takes the count the named subdivision would give
  the same source, so the two OOD splits are size-matched entry by
  entry), `move_weights` the `(flip, subdivide, coarsen)` weights of
  that walk and `mix` the number of extra 2-2 flips per vertex that
  change the local structure at fixed size.

- `MantraDataset(relabel=True)` renames the vertices of every OOD entry
  by a random permutation after the subdivision or walk, so the entry
  leaves the canonical labelling of the release; with
  `division_type="pachner"` alone it is an isomorphic copy of its
  source. `Triangulation.relabel_vertices` does the renaming.

- `MantraDataset` rejects subdivision arguments its `division_type`
  does not take (a typo used to be ignored and yield the test split
  unchanged) and checks a Pachner `target` against `max_vertices` the
  way graded checks `graded_vertex_number`.

- `HasseDiagram` nodes carry their `rank`, so a model can tell the
  ranks apart and a sum readout can count them (as `LeviGraph` nodes
  carry `node_type`).

- `Triangulation2D.move_3_1`, the vertex removal inverse to the 1-3
  move.

- `Triangulation.random_walk`, which applies a random Pachner walk in
  place and returns the triangulation after every `moves_per_step`
  moves.

- `PachnerWalkDataset`, which builds the `MantraDataset` splits and
  expands every train, val and test entry into the snapshots of a
  random Pachner walk (`walk_length`, `moves_per_step`, `move_weights`),
  each with `walk_base` and `walk_step` attributes. The walk
  parameters are encoded in the split file names, so caches do not
  collide with those of `MantraDataset`.

- `AttributeToClassTransform`, `AttributeToRegressionTransform` and
  `NameToClass3MTransform` (with `NAME_TO_CLASS_3M`): stateless task
  transforms whose targets are fixed functions of the stored
  attributes.

  `MANTRADivided` for tuning the on-the-fly balancing
  (`target_count`, `n_moves`, `use_topology_changes`, `max_vertices`,
  `verbose`).

## Removed

- `mantra.datasets.prop_pred`, which imported a `PairwiseSimplicialDS`
  that no longer exists and could not be imported.

- `CreateLabels`, which assigned class indices in encounter order and
  therefore depended on dataset traversal; use
  `AttributeToClassTransform` or a name transform instead.

## Fixed

- `MantraDataset(balanced=True)` balanced nothing since the split
  refactor; the augmented and deduplicated entries are now used.

## Changed

- All sources of the OOD split are drawn before any subdivision runs,
  so every subdivision of one seed starts from the same test entries.
  Graded and partial-stellar OOD splits therefore differ from earlier
  versions at the same seed (their draws used to shift the sources of
  the following classes); barycentric and full-stellar splits are
  unchanged in content. OOD file names carry an `_ss` marker, so older
  caches are rebuilt rather than reused. To reproduce earlier graded or
  partial-stellar results, use 0.0.19.

- `Triangulation2D.random_pachner_move` now samples from the 2-2, 1-3
  and 3-1 moves (`weights` has three entries) and, like the 3D
  version, falls back to the other moves if the chosen one is not
  possible. Previously, random 2D walks only flipped and subdivided,
  so the vertex count never decreased and the walks did not explore
  the full Pachner graph, whose connectivity requires both directions
  of every move. Random sequences for a fixed seed therefore differ
  from earlier versions, including the augmentations produced by
  `balanced=True`.

- `balanced=True` now computes the balanced dataset during `process()`
  via Pachner-move augmentation and deduplication instead of
  downloading a pre-generated release asset. This also fixes the 404
  for recent releases, which no longer shipped balanced assets.

- `balance_dataset` draws augmentations only from original entries
  (never from augmented copies), keeps a random subsample per class
  instead of the smallest triangulations, deduplicates only classes
  that gained augmented entries, and raises informative `ValueError`s
  instead of bare assertions.

- Split caches now also encode `split_proportions` and `stratified`,
  so changing either re-processes instead of silently serving stale
  splits.

- The exact-duplicates warning for oversampled OOD classes now also
  covers stellar subdivision with `fraction=1.0`, which is just as
  deterministic as barycentric subdivision.

## Removed

- `scripts/generate_balanced.py` (superseded by on-the-fly balancing)
  and the `balanced` parameter of the internal dataset URL helper.

# v0.0.16

## Added

- Canonical names for *all* 2-manifolds, thus adding almost 40000 new
  names.

## Fixed

- Renamed `vertex-transitive` attribute to `vertex_transitive`.

- Fixed generation of release notes.

# v0.0.15

## Added

- New deduplication routine for merging different types of manifold
  datasets.

- Added 4787 triangulations of small valence, originally collected by
  Frank Lutz but published *without* homology group information. This
  information was added to the upstream repository and is now part of
  the larger 3-manifold dataset.

# v0.0.14

## Added 

- Proper handling of the dataset versions. In case the provided version does not exist 
  a `ValueError` will be raised and all available versions will be listed to the user. 

## Fixed

- Changed root path to ensure the 2D and 3D datasets do not resolve to the same file. 

# v0.0.13

## Fixed

- Addresses some issues with building the documentation.

# v0.0.12

## Added

- Included `transforms` in main documentation.
- Extended `pyproject.toml` with project URLs.
- Fixed look-and-feel of PyPI documentation.

# v0.0.11

## Fixed

- Minor changes to documentation of individual `transforms`.
- Simplified dependencies.

# v0.0.10

## Fixed

- Minor changes to the release scripts.

# v0.0.9

## Fixed

- Finally using the correct tag by creating a new release *before* we
  send everything to Zenodo. Life is hard.

# v0.0.8

## Fixed

- Using proper logic to name releases on Zenodo. This is just a minor
  choice of aesthetics.

# v0.0.7

## Added

- Depositing the generated files automatically on Zenodo, thus adding
  additional versioning and the ability to refer to a particular version
  using a DOI.

# v0.0.6

## Fixed

- Using better class names to be consistent with the paper.

# v0.0.5 

## Added

- Added automatic upload of the changelog to the GitHub release.

# v0.0.4

## Fixed

- Fixed a tag that prevented fetching the last version of the dataset.
- Added synchronization between the released dataset version and package version on PyPi.
