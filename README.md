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
@unpublished{Schmidt26a,
  title         = {No Triangulation Without Representation: Generalization in Topological Deep Learning},
  author        = {Johannes S. Schmidt and Martin Carrasco and Ernst Röell and Guy Wolf and Nello Blaser and Bastian Rieck},
  year          = 2026,
  eprint        = {2605.06467},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
}
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
here is a simple example, showing the use of *random node features* on the one-skeleton
of a triangulation and how to transform it for using graph neural networks. Note that each *encoding*, in this case `NodeRandomTransform`, does not automatically assign to the `x` feature tensor but creates a tensor with that name. To make the assigned to `x` we need `SelectFeatures` which takes a `src` an `dst` (by default `x`) and a representation (in this case `graph`). More advanced usage can be seen in the examples.

```python
from torch_geometric.transforms import Compose

from mantra.datasets import ManifoldTriangulations
from mantra.transforms import NodeRandomTransform, SelectFeatures
from mantra.representations import OneSkeleton


dataset = ManifoldTriangulations(
    root="./data", # Root of the dataset
    dimension=2, # Dimension of the manifolds in question
    version="latest", # Which version of the dataset to load
    pre_transform=Compose( # Set of transforms to be applied during preprocessing
        [
            OneSkeleton(),
            NodeRandomTransform(), # Assigns random features (default dim=8) on the attribute `random_features`
            SelectFeatures(src="random_features", dst=None, representation="graph"), # Assing `x = random_features`
        ]
    ),
    force_reload=True,
)
```

You can find all the available representations of a traingulation in `mantra.representations`. So far, the supported types are:
1. One Skeleton
2. Dual Graph
3. Hasse Diagram
4. Levi Graph
5. Simplicial Complex (represented with matrices of the boundary operators, Hodge Laplacians and more!)


The dataset also provides an option to "balance" the distribution of homeomorphic types and even the distribution of triangulations. It does
this through Pachner moves and, in 2D, through gluing of manifolds applied to triangulations. An example follows:


```python
from torch_geometric.transforms import Compose

from mantra.datasets import ManifoldTriangulations
from mantra.transforms import NodeRandomTransform, SelectFeatures
from mantra.representations import OneSkeleton


dataset = ManifoldTriangulations(
    root="./data",
    dimension=2,
    version="latest",    
    balanced=True,      # Wether to perform balancing or not, False by default 
    target_count=10,     # Target number of samples per homeomorphic class
    n_moves=5,           # Number of moves to use for adding additional samples
    use_surgery=True,     # If topological surgery, i.e. gluing should be used during augmentation (only 2D)
    seed=0,               # Seed for random sampling of moves
    max_vertices=10,       # Vertex cap for the balancing operation, only applied to new triangulation (None is no cap)
    pre_transform=Compose(
        [
            OneSkeleton(),
            NodeRandomTransform(),
            SelectFeatures(src="random_features", dst=None, representation="graph"),
        ]
    ),
    force_reload=True,
)
```

Additionally you can add a pre-filter to guarantee that all your triangulations have at most $x$ number of vertices.

```python
from torch_geometric.transforms import Compose

from mantra.datasets import ManifoldTriangulations
from mantra.transforms import NodeRandomTransform, SelectFeatures
from mantra.representations import OneSkeleton

dataset = ManifoldTriangulations(
    root="./data",
    dimension=2,
    version="latest",    
    balanced=True,      # Wether to perform balancing or not, False by default 
    target_count=10,     # Target number of samples per homeomorphic class
    n_moves=5,           # Number of moves to use for adding additional samples
    use_surgery=True,     # If topological surgery, i.e. gluing should be used during augmentation (only 2D)
    seed=0,               # Seed for random sampling of moves
    max_vertices=10,       # Vertex cap for the balancing operation, only applied to new triangulation (None is no cap)
    pre_transform=Compose(
        [
            OneSkeleton(),
            NodeRandomTransform(),
            SelectFeatures(src="random_features", dst=None, representation="graph"),
        ]
    ),
    force_reload=True,
)
```


## Using the Dataset - Homeomorphism classification Task

The extended version of MANTRA provides a way to construct ready-to-train versions of the dataset for the homeomorphism type classification task. 
This includes the new out-of-distribution task, all packaged in a single`MantraDataset` class. 

### Simple training split 

The `MantraDataset` class functions similarly to PyG's ZINC dataset. You specify the split you want and it returns the dataset for
that particular split (train, val, test, ood). The first time a fixed configuration of the dataset is called, all the splits are 
constructed and subsequent calls just load from file as long as `force_reload=False`. 

```python
from torch_geometric.transforms import Compose

from mantra.datasets import MantraDataset
from mantra.transforms import NodeRandomTransform, SelectFeatures
from mantra.representations import OneSkeleton

dataset_train =  MantraDataset(
    root="./data",
    dimension=2,
    balanced=False,
    version="latest",
    split_type="train", # Split to load 
    split_proportions=[0.6, 0.2, 0.2], # The split proportion to create
    stratified = False, # Wether to perform stratified splitting, i.e. keep proportions equivalent in each split
    min_sample_per_class = 100, # Minimum amount of samples per homeomorphism type
    seed = 0,
    graded_vertex_number = 20, # Required by the default "graded" division_type: vertex count every OOD sample is grown to
    pre_transform=Compose([
        OneSkeleton(),
        NodeRandomTransform(),
        SelectFeatures(src="random_features", dst=None, representation="graph"),
    ]),
)

dataset_val =  MantraDataset(
    root="./data",
    dimension=2,
    balanced=False,
    version="latest",
    split_type="val", # Split to load 
    split_proportions=[0.6, 0.2, 0.2], # The split proportion to create
    stratified = False, # Wether to perform stratified splitting, i.e. keep proportions equivalent in each split
    min_sample_per_class = 100, # Minimum amount of samples per homeomorphism type
    seed = 0,
    graded_vertex_number = 20,
    pre_transform=Compose([
        OneSkeleton(),
        NodeRandomTransform(),
        SelectFeatures(src="random_features", dst=None, representation="graph"),
    ]),
)

dataset_test =  MantraDataset(
    root="./data",
    dimension=2,
    balanced=False,
    version="latest",
    split_type="test", # Split to load 
    split_proportions=[0.6, 0.2, 0.2], # The split proportion to create
    stratified = False, # Wether to perform stratified splitting, i.e. keep proportions equivalent in each split
    min_sample_per_class = 100, # Minimum amount of samples per homeomorphism type
    seed = 0,
    graded_vertex_number = 20,
    pre_transform=Compose([
        OneSkeleton(),
        NodeRandomTransform(),
        SelectFeatures(src="random_features", dst=None, representation="graph"),
    ]),
)
```
You can now use these for a training pipeline. Balancing, encodings and representations can be combined here as well.

### Training Split + OOD

The out-of-distribution (OOD) task is constructed at the same time the dataset is constructed. By default, it will be empty
```python
from torch_geometric.transforms import Compose

from mantra.datasets import MantraDataset
from mantra.transforms import NodeRandomTransform, SelectFeatures
from mantra.representations import OneSkeleton

dataset_train =  MantraDataset(
    root="./data",
    dimension=2,
    balanced=False,
    version="latest",
    split_type="train", # Split to load 
    split_proportions=[0.6, 0.2, 0.2], # The split proportion to create
    stratified = False, # Wether to perform stratified splitting, i.e. keep proportions equivalent in each split
    min_sample_per_class = 100, # Minimum amount of samples per homeomorphism type
    seed = 0,
    graded_vertex_number = 20, # Required by the default "graded" division_type: vertex count every OOD sample is grown to
    pre_transform=Compose([
        OneSkeleton(),
        NodeRandomTransform(),
        SelectFeatures(src="random_features", dst=None, representation="graph"),
    ]),
)

dataset_val =  MantraDataset(
    root="./data",
    dimension=2,
    balanced=False,
    version="latest",
    split_type="val", # Split to load 
    split_proportions=[0.6, 0.2, 0.2], # The split proportion to create
    stratified = False, # Wether to perform stratified splitting, i.e. keep proportions equivalent in each split
    min_sample_per_class = 100, # Minimum amount of samples per homeomorphism type
    seed = 0,
    graded_vertex_number = 20,
    pre_transform=Compose([
        OneSkeleton(),
        NodeRandomTransform(),
        SelectFeatures(src="random_features", dst=None, representation="graph"),
    ]),
)

dataset_test =  MantraDataset(
    root="./data",
    dimension=2,
    balanced=False,
    version="latest",
    split_type="test", # Split to load 
    split_proportions=[0.6, 0.2, 0.2], # The split proportion to create
    stratified = False, # Wether to perform stratified splitting, i.e. keep proportions equivalent in each split
    min_sample_per_class = 100, # Minimum amount of samples per homeomorphism type
    seed = 0,
    graded_vertex_number = 20,
    pre_transform=Compose([
        OneSkeleton(),
        NodeRandomTransform(),
        SelectFeatures(src="random_features", dst=None, representation="graph"),
    ]),
)
dataset_ood =  MantraDataset(
    root="./data",
    dimension=2,
    balanced=False,
    version="latest",
    split_type="ood", # Split to load 
    split_proportions=[0.6, 0.2, 0.2], # The split proportion to create
    stratified = False, # Wether to perform stratified splitting, i.e. keep proportions equivalent in each split
    min_sample_per_class = 100, # Minimum amount of samples per homeomorphism type
    seed = 0,
    graded_vertex_number = 20,
    pre_transform=Compose([
        OneSkeleton(),
        NodeRandomTransform(),
        SelectFeatures(src="random_features", dst=None, representation="graph"),
    ]),
)
```

## Specifying the task
The main task of MANTRA is predicting the homeomorphism type class of a triangulation, i.e. which manifold it triangulates. However, each triangulation has additional labels related to the properties of the manifold it represents, whichwhich together determine its homeomorphic class. These are a.) orientable, b.) Betti numbers. The former is a binary label that denotes wether the manifold is orientable. The latter is a sequence of integers that count the different dimensional holes.

Task transforms specify the prediction target, while `SelectFeatures` and `SelectAttributes` prepare the final model inputs. For example, the following pipeline creates a Betti-number prediction dataset using node degrees as features:

```python
from torch_geometric.transforms import Compose

from mantra.datasets import ManifoldTriangulations
from mantra.representations import OneSkeleton
from mantra.transforms import (
    NameToClass2MTransform,
    NodeDegreeTransform,
    SelectAttributes,
    SelectFeatures,
)

dataset = ManifoldTriangulations(
    root="./data",
    dimension=2,
    version="latest",
    name="betti_numbers_degree",
    pre_transform=Compose([
        OneSkeleton(),
        NodeDegreeTransform(),
        SelectFeatures(src="degree", dst=None, representation="graph"),
    ]),
    transform=Compose([
        NameToClass2MTransform(),
        SelectAttributes(keep_keys=['x', 'y', 'edge_index', 'n_vertices'])
    ])
)
```

`NameToClass2MTransform` stores the manifold's `name` (homeomorphism type) encoded as an integer in `data.y`. `SelectFeatures` assigns the computed degree values to `data.x`, and `SelectAttributes` keeps only the specified attributes, in this case `x`, `y`, `edge_index` and `n_vertices`.

All task transforms are *stateless*: the target of a sample is a pure function of its stored attributes, so it never depends on the order in which samples are visited or on the subset that is loaded. Class indices come from fixed mappings: `NAME_TO_CLASS_2M` and `NAME_TO_CLASS_3M` (exported from `mantra.transforms`) for the homeomorphism types, used by the shorthands `NameToClass2MTransform` and `NameToClass3MTransform`, or a `{value: index}` mapping passed to `AttributeToClassTransform(source, mapping)` for any other attribute. For integer-valued attributes such as `genus`, build that mapping once from the values present in the *full* dataset, not from a split. `AttributeToRegressionTransform(source)` converts a scalar or fixed-length attribute to a float target of shape `(1, k)`. Remapping canonical class indices to a contiguous range over the classes present in a training split needs to be performed in the training code.

```python
from mantra.transforms import (
    NAME_TO_CLASS_2M,
    AttributeToClassTransform,
    AttributeToRegressionTransform,
)

# Class index from a fixed mapping (NameToClass2MTransform() is the
# shorthand for exactly this).
AttributeToClassTransform("name", mapping=NAME_TO_CLASS_2M)

# Integer-valued attribute: build the mapping once from the values
# present in the full dataset, so that it is independent of the split.
genus_values = sorted({int(data.genus) for data in dataset})
AttributeToClassTransform(
    "genus", mapping={v: i for i, v in enumerate(genus_values)}
)

# Float regression target of shape (1, 1) from a scalar attribute.
AttributeToRegressionTransform("genus")
```

Node-level targets are supported by `AttributeToNodeRegressionTransform(source, mask_first=False)` and `AttributeToNodeClassTransform(source, mapping, mask_first=False)`, which turn an attribute holding one tensor value per vertex (e.g. the second Chern class paired with each toric divisor of a Calabi-Yau triangulation) into `data.y` of shape `(n_vertices, 1)` (float regression target) or `(n_vertices,)` (class indices from a fixed `mapping`). Both also store a boolean `data.node_mask` selecting the supervised vertices; with `mask_first=True` the first vertex (e.g. the origin of the polytope, which carries no target) is excluded.

```python
from mantra.transforms import AttributeToNodeRegressionTransform

# Regress one value per vertex, ignoring the first vertex in the loss.
AttributeToNodeRegressionTransform("c2", mask_first=True)
```

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
A: The core class of the MANTRA package is `ManifoldTriangulations`. `ManifoldTriangulations` allows the user to load the MANTRA dataset using a `InMemoryDataset` format from [`torch_geometric`]([`torch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/)). Additionally, the `MantraDataset` class allows loading a dataset split.

#### Q: What representations are available in this dataset?
A: The available representations are:

| Type | Representations |
| --- | --- |
| Graph | `OneSkeleton`, `DualGraph`, `HasseDiagram`, `LeviGraph` |
| Simplicial complex | `IncidenceSimplicialComplex`, `AdjacencySimplicialComplex`, `CoadjacencySimplicialComplex`, `UpLaplacianSimplicialComplex`, `DownLaplacianSimplicialComplex`, `HodgeLaplacianSimplicialComplex` |

#### Q: What encodings are available in this dataset?
A: `NodeRandomTransform` assigns random feature vectors to graph nodes, while `SimplexRandomTransform` assigns them to simplices of a chosen dimension. `NodeDegreeTransform` uses each graph node's degree as its feature. `MomentCurveEmbedding` gives vertices canonical coordinates on a moment curve, with optional rotation and normalization. `EffectiveResistanceEmbedding` computes effective-resistance features for simplices from incidence matrices, and `EffectiveResistanceStatisticsEmbedding` summarizes these features using statistics such as the mean, standard deviation, extrema, median, and quartiles.

#### Q: What tasks are available in this dataset?
A: MANTRA supports **homeomorphism-type classification**, where a model predicts the manifold represented by a triangulation; **orientability classification**, a binary task that determines whether a manifold is orientable; and **Betti-number prediction**, which predicts the ranks of the manifold's homology groups. 

*Have a question that's not answered here? Please open an issue on our GitHub repository.*

# Acknowledgements

This work is dedicated to [Frank H. Lutz](https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/),
who passed away unexpectedly on November 10, 2023. May his memory be
a blessing.
