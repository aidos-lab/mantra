## Installation and getting the Dataset

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

