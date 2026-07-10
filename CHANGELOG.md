# CHANGELOG

This is a [changelog](https://keepachangelog.com/) of all notable
changes to this project. We adhere to [Semantic Versioning](https://semver.org/).

# Unreleased

## Added

- `balance_kwargs` parameter on `ManifoldTriangulations` and
  `MANTRADivided` for tuning the on-the-fly balancing
  (`target_count`, `n_moves`, `use_topology_changes`, `max_vertices`,
  `verbose`).

## Changed

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
