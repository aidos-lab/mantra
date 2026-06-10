# MANTRA: The Manifold Triangulations Assemblage

[![arXiv](https://img.shields.io/badge/arXiv-2410.02392-b31b1b.svg)](https://arxiv.org/abs/2410.02392)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14103581.svg)](https://doi.org/10.5281/zenodo.14103581) [![Maintainability](https://qlty.sh/badges/88ae05e7-c892-4edf-9dff-38cda745593f/maintainability.svg)](https://qlty.sh/gh/aidos-lab/projects/mantra) [![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/MANTRA)](https://github.com/aidos-lab/MANTRA/graphs/contributors) [![CHANGELOG](https://img.shields.io/badge/Changelog--default)](https://github.com/aidos-lab/mantra/blob/main/CHANGELOG.md) [![License](https://img.shields.io/github/license/aidos-lab/MANTRA)](/LICENSE.md)

![image](https://github.com/aidos-lab/mantra/blob/main/_static/manifold_triangulation_orbit.gif)

MANTRA is a dataset consisting of *combinatorial triangulations* of
manifolds. It can be used to create novel algorithms in topological
deep learning or debug existing ones. See our [ICLR 2025
paper](https://openreview.net/pdf?id=X6y5CC44HM) for more details and
our [benchmarks repository](https://github.com/aidos-lab/mantra-benchmarks) for
additional code to reproduce all experiments.

Please use the following citation for our work:

```bibtex
@inproceedings{Ballester25a,
  title         = {{MANTRA}: {T}he {M}anifold {T}riangulations {A}ssemblage},
  author        = {Rubén Ballester and Ernst Röell and Daniel Bīn Schmid and Mathieu Alain and Sergio Escalera and Carles Casacuberta and Bastian Rieck},
  year          = 2025,
  booktitle     = {International Conference on Learning Representations},
  url           = {https://openreview.net/forum?id=X6y5CC44HM},
}
```

## Getting the Dataset

The raw MANTRA dataset consisting of $2$- and $3$-manifolds with up to $10$ vertices 
is provided [here](https://github.com/aidos-lab/mantra/releases/latest). 
For machine-learning applications and research, we provide a custom
dataset loader package, which can be installed via the following command:

```console
pip install mantra-dataset
```

After installation, the dataset can be used like this:

```python
from mantra.datasets import ManifoldTriangulations

dataset = ManifoldTriangulations(
    root="./data",      # root folder for storing data
    dimension=2,        # Whether to load 2- or 3-manifolds
    version="latest"    # Which version of the dataset to load
)
```

Provided you have [`pytorch-geometric`](https://github.com/pyg-team/pytorch_geometric) installed,
here is a more comprehensive example, showing the use of *random node features* and how to transform it
for using graph neural networks:

```python
from torch_geometric.transforms import Compose
from torch_geometric.transforms import FaceToEdge

from mantra.datasets import ManifoldTriangulations
from mantra.transforms import NodeIndex
from mantra.transforms import RandomNodeFeatures


dataset = ManifoldTriangulations(
    root="./data",
    dimension=2,
    version="latest",
    transform=Compose(
        [
            NodeIndex(),
            RandomNodeFeatures(),
            # Converts face indices to edge indices, thus essentially
            # making the 1-skeleton available to a model.
            FaceToEdge(remove_faces=False),
        ]
    ),
    force_reload=True,
)
```

## Dataset variants and model bundles

### Subdivided variants

In addition to the original triangulations, MANTRA can be loaded at increasing
levels of **barycentric subdivision**. Each subdivision refines every
top-dimensional simplex, raising the vertex/simplex counts while preserving the
homeomorphism type (and hence the label). This yields a controlled "resolution"
axis that is useful for studying how higher-order models scale with complex
size.

`ManifoldTriangulations` understands a `subdivision_level` argument and the
release-artifact naming `{dim}_manifolds{_balanced}_bary{L}.json[.gz]`:

```python
from mantra.datasets import ManifoldTriangulations

ds = ManifoldTriangulations(
    root="./data", dimension=2, balanced=True,
    subdivision_level=2,        # loads 2_manifolds_balanced_bary2.json(.gz)
)
```

Subdivided files are produced with the bundled generator. Integer levels of
`full_barycentric`/`stellar` are written under the exact canonical name above,
so they are release-ready (upload as `.json.gz` next to the base release and
the loader will auto-download them):

```console
python -m mantra.generate_subdivided_datasets \
    --input 2_manifolds.json --output-dir ./subdivided \
    --dimension 2 --n-levels 2 --mode full_barycentric \
    --min-class-count 0 --n-smallest 0
# -> ./subdivided/2_manifolds_bary1.json, 2_manifolds_bary2.json
```

| Flag | Meaning |
| --- | --- |
| `--input` / `--output-dir` | Source raw JSON / output directory. |
| `--dimension` / `--balanced` | Build the canonical output filename. |
| `--mode` | `full_barycentric`, `stellar`, or `graded`. |
| `--n-levels` | Integer #levels; a fraction in `(0,1)` for partial `stellar`; the target vertex count for `graded`. |
| `--min-vertices` | Drop subdivided entries below this vertex count. |
| `--min-class-count` | Pre-filter: drop classes with `<=` this many entries (`0` disables). |
| `--n-smallest` | Keep only the N smallest valid samples per class (`0` disables). |
| `--prefix` | Descriptive name for fractional/`graded` outputs (consumed via `local_path`). |

Fractional (partial `stellar`) and `graded` outputs cannot be expressed by an
integer `subdivision_level`; they use a descriptive `{prefix}` name and are
loaded via the `local_path` argument instead.

### Model bundles

`make_dataset` produces a dataset whose `pre_transform` is **ready to train** a
given model. Graph models pick a single `representation`; the higher-order
simplicial-complex models use a `bundle`, which stacks the exact ordered
connectivity transforms each one expects and lands per-rank features on
`x_0..x_d`:

```python
from mantra.datasets import make_dataset

# Higher-order (simplicial complex) — features on x_0..x_d:
ds = make_dataset(root="./data", dimension=2, bundle="cwn", featurization="random")

# Graph models — features on x:
ds = make_dataset(root="./data", dimension=2, representation="one_skeleton",
                  featurization="degree")
```

Pass **exactly one** of `representation` or `bundle`.

| Bundle | Connectivity stack (all `signed`, `propagate=True`) | `select` |
| --- | --- | --- |
| `cwn` | `incidence`, `adjacency` | `sc` |
| `sccnn` | `incidence`, `up_laplacian`, `down_laplacian`, `hodge_laplacian` | `sc` |
| `san` | `incidence`, `up_laplacian`, `down_laplacian` | `sc` |

Graph models such as **gcn** are not bundles — use
`representation=` (`one_skeleton`, `dual`, `hasse`, `triangulation`).

**Featurization compatibility.** Graph `representation` paths accept
`random`, `node_degree`, `mc`, and `rw`. Model `bundle`s accept only `random`
and `mc` (the two embeddings that propagate to every rank); `node_degree`/`rw`
read `edge_index` and are graph-only.

## More Examples 

Please find more example notebooks in the [`examples`](/examples)
folder:

1. [Adding new tasks to MANTRA](/examples/adding_new_task.ipynb)
2. [Training a GNN on MANTRA](/examples/train_gnn.ipynb)
3. [Visualizing the MANTRA dataset](/examples/visualize_data.ipynb)

## FAQ

#### Q: Why MANTRA?
A: MANTRA is one of the first datasets providing prediction tasks that provably depend on the high-order features of the input data, in the case of MANTRA, simplices. MANTRA contributes to the benchmarking ecosystem for high-order neural networks by providing a large set of triangulations with precomputed topological properties that can be easily computed with deterministic algorithms but that are hard to compute for predictive models. The topological properties contained in MANTRA are elementary, meaning that good networks tackling complex topological problems should be able to completely solve this dataset. Currently, there is no model that can solve all the prediction tasks proposed in the dataset's paper. 

#### Q: Why topological features?
A: Topology forms a fundamental theoretical foundation for natural sciences like physics and biology. Understanding a system's topology often reveals critical insights hardly accessible through other analytical methods. For neural networks to effectively tackle problems in these domains, they must develop capabilities to leverage topological information. This requires network architectures capable of identifying basic topological invariants in data—precisely the invariants that MANTRA provides. By incorporating these topological features, neural networks can capture essential structural and relational properties that traditional approaches might miss, enhancing their ability to model complex natural phenomena.


#### Q: Which are the main functions and classes implemented in this dataset?
A: The core class of the MANTRA package is `ManifoldTriangulations`. `ManifoldTriangulations` allows the user to load the MANTRA dataset using a `InMemoryDataset` format from [`torch_geometric`]([`torch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/)). The transforms `NodeIndex`, `RandomNodeFeatures`, `DegreeTransform`, and `DegreeTransformOneHot`are also provided in this package. Concretely, `NodeIndex` transforms the original triangulation format in a torch-like tensor, and `RandomNodeFeatures`, `DegreeTransform`, and `DegreeTransformOneHot` assign input feature vectors to vertices in a the `x` attribute of the input `Data` representing a triangulation based either on random features or on the degree of each vertex, respectively.

*Have a question that's not answered here? Please open an issue on our GitHub repository.*

# Acknowledgements

This work is dedicated to [Frank H. Lutz](https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/),
who passed away unexpectedly on November 10, 2023. May his memory be
a blessing.
