# MANTRA: The Manifold Triangulations Assemblage


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14103581.svg)](https://doi.org/10.5281/zenodo.14103581) [![Maintainability](https://api.codeclimate.com/v1/badges/82f86d7e2f0aae342055/maintainability)](https://codeclimate.com/github/aidos-lab/MANTRA/maintainability) ![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/MANTRA) [![CHANGELOG](https://img.shields.io/badge/Changelog--default)](https://github.com/aidos-lab/mantra/blob/main/CHANGELOG.md) ![License](https://img.shields.io/github/license/aidos-lab/MANTRA) 

![image](_static/manifold_triangulation_orbit.gif)

MANTRA is a dataset consisting of *combinatorial triangulations* of
manifolds. It can be used to create novel algorithms in topological
deep learning or debug existing ones. See our [ICLR 2025 paper](https://openreview.net/pdf?id=X6y5CC44HM) for more details on potential experiments.

Please use the following citation for our work:

```bibtex
@inproceedings{
ballester2025mantra,
title={{MANTRA}: {T}he {M}anifold {T}riangulations {A}ssemblage},
author={Rub{\'e}n Ballester and Ernst R{\"o}ell and Daniel Bin Schmid and Mathieu Alain and Sergio Escalera and Carles Casacuberta and Bastian Rieck},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=X6y5CC44HM}
}
```

## Getting the Dataset

The raw MANTRA dataset consisting of $2$- and $3$-manifolds with up to $10$ vertices 
is provided [here](https://github.com/aidos-lab/mantra/releases/latest). 
For machine-learning applications and research, we provide a custom
dataset loader, which can be installed via the following command:

```python
pip install mantra-dataset
```

After installation, the dataset can be used like this:

```python
from mantra.datasets import ManifoldTriangulations

dataset = ManifoldTriangulations(root="./data", manifold="2", version="latest")
```

To add random node features to the dataset, we can add it as a transform to the dataset.

```python
# Load all required packages. 
from torch_geometric.transforms import Compose, FaceToEdge

# Load the mantra dataset
from mantra.datasets import ManifoldTriangulations
from mantra.transforms import NodeIndex, RandomNodeFeatures

dataset = ManifoldTriangulations(root="./data", manifold="2", version="latest",
                                 transform=Compose([
                                        NodeIndex(),
                                        RandomNodeFeatures(),
                                        FaceToEdge(remove_faces=False),
                                        ]
                                    ),
                                    force_reload=True,
                                )

```

## Examples 

Under the `examples` folder we have included three notebooks. The first notebook 
contains an example for training a Graph Neural Network with the MANTRA dataset, the second notebook contains an analysis of the data distribution, and the third one an 
example on how to extend current labels to develop new prediction tasks.


## Acknowledgements

This work is dedicated to [Frank H. Lutz](https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/),
who passed away unexpectedly on November 10, 2023. May his memory be
a blessing.

## FAQ

#### Q: Why MANTRA?
A: MANTRA is one of the first datasets providing prediction tasks that provably depend on the high-order features of the input data, in the case of MANTRA, simplices. MANTRA contributes to the benchmarking ecosystem for high-order neural networks by providing a large set of triangulations with precomputed topological properties that can be easily computed with deterministic algorithms but that are hard to compute for predictive models. The topological properties contained in MANTRA are elementary, meaning that good networks tackling complex topological problems should be able to completely solve this dataset. Currently, there is no model that can solve all the prediction tasks proposed in the dataset's paper. 

#### Q: Why topological features?
A: Topology forms a fundamental theoretical foundation for natural sciences like physics and biology. Understanding a system's topology often reveals critical insights hardly accessible through other analytical methods. For neural networks to effectively tackle problems in these domains, they must develop capabilities to leverage topological information. This requires network architectures capable of identifying basic topological invariants in data—precisely the invariants that MANTRA provides. By incorporating these topological features, neural networks can capture essential structural and relational properties that traditional approaches might miss, enhancing their ability to model complex natural phenomena.


#### Q: Which are the main functions and classes implemented in this dataset?
A: The core class of the MANTRA package is `ManifoldTriangulations`. `ManifoldTriangulations` allows the user to load the MANTRA dataset using a `InMemoryDataset` format from [`torch_geometric`]([`torch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/)). The transforms `NodeIndex`, `RandomNodeFeatures`, `DegreeTransform`, and `DegreeTransformOneHot`are also provided in this package. Concretely, `NodeIndex` transforms the original triangulation format in a torch-like tensor, and `RandomNodeFeatures`, `DegreeTransform`, and `DegreeTransformOneHot` assign input feature vectors to vertices in a the `x` attribute of the input `Data` representing a triangulation based either on random features or on the degree of each vertex, respectively.

*Have a question that's not answered here? Please open an issue on our GitHub repository.*

