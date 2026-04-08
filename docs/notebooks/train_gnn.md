# Training a GNN on the Mantra Dataset

In this tutorial, we provide an example use-case for the mantra dataset. We show 
how to train a GNN to predict the orientability based on random node features. 

The `torch-geometric` interface for the MANTRA dataset can be installed  with 
pip via the command 
```{python}
pip install mantra-dataset
```

As a preprocessing step we apply three transforms to the base dataset.
Since the dataset does not have intrinsic coordinates attached to the vertices, 
we first have to create a transform that generates random node features.
Each manifold in MANTRA comes as a list of triples, where the integers in each 
triple are vertex id's. The starting id in each manifold is $1$ and has to be 
converted to a torch-geometric compliant $0$-based index.
GNN's are typically trained on graphs and the FaceToEdge transform converts our
manifold to a graph. 

For each of the transforms we use a single class and are succesively applied to
form the final transformed dataset. 


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

# Instantiate the dataset. Following the `torch-geometric` API, we download the 
# dataset into the root directory. 
dataset = ManifoldTriangulations(root="./data", dimension=2, version="latest",
                                 transform=Compose([
                                        NodeIndex(),
                                        RandomNodeFeatures(),
                                        FaceToEdge(remove_faces=True),
                                        ]
                                    )
                                )

```


```python
train_dataset, test_dataset = random_split(
            dataset,
            [0.8,0.2
            ],
        )  # type: ignore

train_dataloader = DataLoader(train_dataset,batch_size=32)
test_dataloader = DataLoader(test_dataset,batch_size=32)

```


```python
class GCN(nn.Module):
    ''' 
    A standard Graph Convolutional Neural Network with three layers for 
    predicting the orientability of the manifold. 
    Note that since our model only uses the edge information, it is not 
    able to learn the orientability. It will therefore only serve as 
    an example. 
    '''
    def __init__(self):
        super().__init__()

        self.conv_input = GCNConv(
           8, 16
        )
        self.final_linear = nn.Linear(
            16, 1
        )

    def forward(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        
        # 1. Obtain node embeddings
        x = self.conv_input(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.final_linear(x)
        return x
```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = nn.BCEWithLogitsLoss()

model.train()
for epoch in range(10):
    for batch in train_dataloader: 
        batch.orientable = batch.orientable.to(torch.float)
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out.squeeze(), batch.orientable)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, {loss.item()}")

```

    Epoch 0, 0.10134144872426987
    Epoch 1, 0.09631027281284332
    Epoch 2, 0.08791007101535797
    Epoch 3, 0.08849794417619705
    Epoch 4, 0.08793244510889053
    Epoch 5, 0.08987350761890411
    Epoch 6, 0.08849993348121643
    Epoch 7, 0.08913585543632507
    Epoch 8, 0.08902224153280258
    Epoch 9, 0.09069301933050156



```python
correct = 0
total = 0
model.eval()
for testbatch in test_dataloader: 
    testbatch.to(device)
    pred = model(testbatch)
    correct += ((pred.squeeze() < 0) == testbatch.orientable).sum()
    total += len(testbatch)

acc = int(correct) / int(total)
print(f'Accuracy: {acc:.4f}')
```

    Accuracy: 0.0800

