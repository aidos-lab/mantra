# MANTRA: Manifold Triangulations Assembly

[![Maintainability](https://api.codeclimate.com/v1/badges/82f86d7e2f0aae342055/maintainability)](https://codeclimate.com/github/aidos-lab/MANTRA/maintainability) ![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/MANTRA) ![GitHub](https://img.shields.io/github/license/aidos-lab/MANTRA) 

![image](_static/manifold_triangulation_orbit.gif)

Please use the following citation for our work:

```bibtex
@misc{ballester2024mantramanifoldtriangulationsassemblage,
      title={ {MANTRA}: {T}he {M}anifold {T}riangulations {A}ssemblage}, 
      author={Rub{\'e}n Ballester and Ernst R{\"o}ell and Daniel Bin Schmid and Mathieu Alain and Sergio Escalera and Carles Casacuberta and Bastian Rieck},
      year={2024},
      eprint={2410.02392},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.02392}, 
}
```

## Getting the Dataset

The raw MANTRA dataset consisting of the $2$ and $3$ manifolds with up to $10$ vertices 
is provided [here](https://github.com/aidos-lab/mantra/releases/latest). 
For machine learning applications and research, we provide a custom [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/stable/) dataset in the form of a python package. 

For installations via pip, run  

The raw datasets, consisting of the 2 and 3 manifolds with up to 10
vertices, can be manually downloaded 
[here](https://github.com/aidos-lab/mantra/releases/latest). 
A pytorch geometric wrapper for the dataset is installable via the following 
command.

```python
pip install mantra-dataset
```

After installation the dataset can be used with the following snippet.

```python
from mantra.datasets import ManifoldTriangulations

dataset = ManifoldTriangulations(root="./data", manifold="2", version="latest")
```

## Acknowledgments

This work is dedicated to [Frank H. Lutz](https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/),
who passed away unexpectedly on November 10, 2023. May his memory be
a blessing.
