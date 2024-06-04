# MANTRADataset

# Structure

**mantra** Contains the code for the dataset definition for PyTorch Geometric.

**dataprocessing** Contains the code to publish the dataset to a github release, including the preprocessing scripts. NOTE: We might want to move this to another folder, or rename.

## Dataset

The raw datasets, consisting of the 2 and 3 manifolds with up to 10 vertices, can be downloaded under releases. A pytorch geometric wrapper for the dataset is installable via the following command.

```{python}
pip install "git+https://github.com/aidos-lab/MANTRADataset/#subdirectory=mantra"
```

After installation the dataset can be used with the follwing snippet.

```{python}
from mantra.simplicial import SimplicialDataset

dataset = SimplicialDataset(root="./data", manifold="2")
```

**Warning** Since the repository is private, the dataset can not download the data from the github release due to access restrictions. Hence one has to manually download the "2_manifolds.json.gz" or "3_manifolds.json.gz" to the raw folder for the code to run correctly.
