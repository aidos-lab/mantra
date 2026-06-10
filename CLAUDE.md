# MANTRA — Manifold Triangulations Assemblage

## Project overview
Dataset library for combinatorial triangulations of manifolds (2D surfaces and 3D manifolds), built on PyTorch Geometric's `InMemoryDataset`. Used for topological deep learning research — manifold triangulation classification.

## Related projects
- `../mantra-pp-benchmarks/` — Models, data loaders, experiments (GNNs, simplicial/cell complex models). Uses this package as a dependency (`mantra-dataset`).

## Architecture
- `mantra/datasets/base.py` — `ManifoldTriangulations(InMemoryDataset)`: downloads JSON.gz from GitHub releases, processes into PyG `Data` objects
- `mantra/datasets/pairwise.py` — `PairwiseSimplicialDS`: pairwise comparison dataset
- `mantra/representations/` — Triangulation representations (dual graph, 1-skeleton, Hasse diagram, simplicial connectivity matrices)
- `mantra/representations/internal/` — `Simplex`, `SimplexTrie` data structures
- `mantra/transforms/` — Moment curve embedding, effective resistance, feature selection, label creation
- `mantra/deduplication.py` — Isomorphic duplicate detection (f-vector + degree seq + edge-simplex-count → WL hash → VF2 on incidence graph)
- `mantra/lex_to_json.py` — Converts raw lexicographical triangulation format to JSON
- `mantra/manifold_types.py` — Enum for 2-manifold and 3-manifold types

## Data format
- Raw: JSON with fields `id`, `triangulation` (list of top-level simplices, 1-indexed vertices), `dimension`, `n_vertices`, `betti_numbers`, `name`, etc.
- Triangulations are stored as nested lists: `[[1,2,3,4], [1,2,3,5], ...]` where each inner list is a top-simplex
- For closed 3-manifolds: `E = V + T`, `F = 2T` (Euler characteristic + double counting), so f-vector is determined by `(V, T)`

## Environment
- Python environment: `micromamba`, env name `uv` at `/Users/schmidjo/mamba/envs/uv/`
- Run commands with: `/Users/schmidjo/mamba/envs/uv/bin/python`
- Also used for the sibling `mantra-pp-benchmarks` repo
- If a package is missing, install it into this env rather than switching envs
- Key deps: `torch`, `torch_geometric`, `networkx`, `scipy`, `numpy`, `requests`
- Formatting: `black` (line-length 79), `ruff`

## Testing
- `pytest test/` — unit tests
- `pytest test/test_deduplication.py --dataset-path 3_manifolds.json` — integration dedup test (requires dataset file)
- Test config (conftest) uses `--dataset-path` custom option for dataset integration tests

## Conventions
- 1-indexed vertices in JSON/raw data
- Docstrings follow numpy style
- Code formatted with black (79 chars) and ruff
