# MANTRA: Manifold Triangulations Assembly

[![Maintainability](https://api.codeclimate.com/v1/badges/82f86d7e2f0aae342055/maintainability)](https://codeclimate.com/github/aidos-lab/MANTRA/maintainability) ![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/MANTRA) ![GitHub](https://img.shields.io/github/license/aidos-lab/MANTRA) 

## Getting the Dataset

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
