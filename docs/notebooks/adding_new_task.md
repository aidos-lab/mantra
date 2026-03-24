# Adding new tasks in MANTRA

While we released MANTRA together with a set of predefined prediction tasks, MANTRA can actually be used to benchmark neural networks on other interesting topological tasks involving triangulations, such as Euler characteristic prediction or synthetic generation of triangulations. In this tutorial, we'll show you how to extend the base library of MANTRA to consider Euler characteristics of the input values as labels. The process is straightforward and opens up new research possibilities!

The first step is to generate the Euler characteristic for each input manifold. To do this, we create a new transform that assigns to each input its respective Euler characteristic, formatting this number with a torch.Tensor type.


```python
import torch 

class EulerCharacteristic:
    def __call__(self, data):
        data.euler_characteristic = torch.tensor([sum([((-1)**i)*betti_number for i, betti_number in enumerate(data.betti_numbers)])])
        return data
```

Now, we simply add the transform to the dataset initialization. We made MANTRA to be compatible with the `torch_geometric` dataset API, so you can use transforms, filters, and force reaload.


```python
from mantra.datasets import ManifoldTriangulations
from mantra.transforms import NodeIndex, RandomNodeFeatures
from torch_geometric.transforms import Compose, FaceToEdge

dataset = ManifoldTriangulations(root="./data", dimension=2, version="latest", 
                                 transform=EulerCharacteristic)
```

Now, let's train a simple GNN to predict the Euler characteristics. For more information on training models in MANTRA, see the `train_gnn.ipynb` notebook.


```python
# Load all required packages. 
import torch 
import torch.nn.functional as F
from torch import nn
from torch.utils.data import random_split

from torchvision.transforms import Compose

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, FaceToEdge

from torch_geometric.nn import GCNConv, global_mean_pool

# Load the mantra dataset
from mantra.datasets import ManifoldTriangulations
from mantra.transforms import NodeIndex, RandomNodeFeatures

dataset = ManifoldTriangulations(root="./data", dimension=2, version="latest",
                                 transform=Compose([
                                        NodeIndex(),
                                        RandomNodeFeatures(),
                                        FaceToEdge(remove_faces=True),
                                        EulerCharacteristic()
                                        ]
                                    )
                                )

train_dataset, test_dataset = random_split(
            dataset,
            [0.8,0.2],
        )

train_dataloader = DataLoader(train_dataset,batch_size=32)
test_dataloader = DataLoader(test_dataset,batch_size=32)
```


```python
class GCN(nn.Module):
    ''' 
    A standard Graph Convolutional Neural Network with three layers for 
    predicting the Euler characteristic of the manifold. 
    '''
    def __init__(self):
        super().__init__()
        self.conv_input = GCNConv(8, 16)
        self.final_linear = nn.Linear(16, 1)

    def forward(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        return self.final_linear(F.dropout(global_mean_pool(self.conv_input(x, edge_index), batch), p=0.5, training=self.training))

```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(10):
    for batch in train_dataloader:
        batch.euler_characteristic = batch.euler_characteristic.to(torch.float)
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out.squeeze(), batch.euler_characteristic)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, {loss.item()}")

```

    Epoch 0, 0.6587832570075989
    Epoch 1, 0.6587832570075989
    Epoch 2, 0.6926756501197815
    Epoch 3, 0.6598934531211853
    Epoch 4, 0.6575506329536438
    Epoch 5, 0.6574848890304565
    Epoch 6, 0.6762682199478149
    Epoch 7, 0.6555275321006775
    Epoch 8, 0.6584398746490479
    Epoch 9, 0.6660548448562622

