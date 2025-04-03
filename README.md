# MANTRA: The Manifold Triangulations Assemblage


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14103581.svg)](https://doi.org/10.5281/zenodo.14103581) [![Maintainability](https://api.codeclimate.com/v1/badges/82f86d7e2f0aae342055/maintainability)](https://codeclimate.com/github/aidos-lab/MANTRA/maintainability) ![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/MANTRA) [![CHANGELOG](https://img.shields.io/badge/Changelog--default)](https://github.com/aidos-lab/mantra/blob/main/CHANGELOG.md) ![License](https://img.shields.io/github/license/aidos-lab/MANTRA) 

![image](_static/manifold_triangulation_orbit.gif)

MANTRA is a dataset consisting of *combinatorial triangulations* of
manifolds. It can be used to create novel algorithms in topological
deep learning or debug existing ones. See our [ICLR 2025 paper](https://openreview.net/pdf?id=X6y5CC44HM) for more details on potential experiments.

Please use the following citation for our work:

```bibtex
@inproceedings{Ballester25a,
  title         = {{MANTRA}: {T}he {M}anifold {T}riangulations {A}ssemblage},
  author        = {Rubén Ballester and Ernst Röell and Daniel Bīn Schmid and Mathieu Alain and Sergio Escalera and Carles Casacuberta and Bastian Rieck},
  year          = 2025,
  eprint        = {2410.02392},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  booktitle     = {International Conference on Learning Representations},
  url           = {https://openreview.net/forum?id=X6y5CC44HM},
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

Under the `examples` folder we have included two notebooks. The first notebook 
contains an example for training a Graph Neural Network with the MANTRA dataset while  
the second notebook contains an analysis of the data distribution.


## Acknowledgements

This work is dedicated to [Frank H. Lutz](https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/),
who passed away unexpectedly on November 10, 2023. May his memory be
a blessing.
