# MANTRA: Manifold Triangulations Assembly

[![Maintainability](https://api.codeclimate.com/v1/badges/82f86d7e2f0aae342055/maintainability)](https://codeclimate.com/github/aidos-lab/MANTRA/maintainability) ![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/MANTRA) ![GitHub](https://img.shields.io/github/license/aidos-lab/MANTRA)

## Getting the Dataset

The raw datasets, consisting of the 2 and 3 manifolds with up to 10
vertices, can be downloaded under releases. A pytorch geometric wrapper
for the dataset is installable via the following command.

```{python}
pip install "git+https://github.com/aidos-lab/MANTRADataset/#subdirectory=mantra"
```

After installation the dataset can be used with the follwing snippet.

```{python}
from mantra.simplicial import SimplicialDataset

dataset = SimplicialDataset(root="./data", manifold="2")
```

## Folder Structure

## Data Format

> [!NOTE]
> This section is mostly *information-oriented* and provides a brief
> overview of the data format, followed by a short [example](#example).

Each dataset consists of a list of triangulations, with each
triangulation having the following attributes:

* `id` (required, `str`): This attribute refers to the original ID of
  the triangulation as used by the creator of the dataset (see
  [below](#acknowledgments)). This facilitates comparisons to the
  original dataset if necessary.

* `triangulation` (required, `list` of `list` of `int`): A doubly-nested
  list of the top-level simplices of the triangulation.

* `n_vertices` (required, `int`): The number of vertices in the
  triangulation. This is **not** the number of simplices.

* `name` (required, `str`): A canonical name of the triangulation, such
  as `S^2` for the two-dimensional [sphere](https://en.wikipedia.org/wiki/N-sphere).
  If no canonical name exists, we store an empty string.

* `betti_numbers` (required, `list` of `int`): A list of the [Betti
  numbers](https://en.wikipedia.org/wiki/Betti_number) of the
  triangulation, computed using $Z_2$ coefficients. This implies that
  [torsion](https://en.wikipedia.org/wiki/Homology_(mathematics))
  coefficients are stored in another attribute.

* `torsion_coefficients` (required, `list` of `str`): A list of the
  [torsion
  coefficients](https://en.wikipedia.org/wiki/Homology_(mathematics)) of
  the triangulation. An empty string `""` indicates that no torsion
  coefficients are available in that dimension. Otherwise, the original
  spelling of torsion coefficients is retained, so a valid entry might
  be `"Z_2"`. 

* `genus` (optional, `int`): For 2-manifolds, contains the
  [genus](https://en.wikipedia.org/wiki/Genus_(mathematics)) of the
  triangulation.

* `orientable` (optional, `bool`): Specifies whether the triangulation
  is [orientable](https://en.wikipedia.org/wiki/Orientability) or not.

### Example

```json
[
  {
    "id": "manifold_2_4_1",
    "triangulation": [
      [1,2,3],
      [1,2,4],
      [1,3,4],
      [2,3,4]
    ],
    "dimension": 2,
    "n_vertices": 4,
    "betti_numbers": [
      1,
      0,
      1
    ],
    "torsion_coefficients": [
      "",
      "",
      ""
    ],
    "name": "S^2",
    "genus": 0,
    "orientable": true
  },
  {
    "id": "manifold_2_5_1",
    "triangulation": [
      [1,2,3],
      [1,2,4],
      [1,3,5],
      [1,4,5],
      [2,3,4],
      [3,4,5]
    ],
    "dimension": 2,
    "n_vertices": 5,
    "betti_numbers": [
      1,
      0,
      1
    ],
    "torsion_coefficients": [
      "",
      "",
      ""
    ],
    "name": "S^2",
    "genus": 0,
    "orientable": true
  }
]
```

### Design Decisions

> [!NOTE]
> This section is *understanding-oriented* and provides additional
> justifications for our data format.

The datasets are converted from their original (mixed) lexicographical
format. A triangulation in lexicographical format could look like this:

```
manifold_lex_d2_n6_#1=[[1,2,3],[1,2,4],[1,3,4],[2,3,5],[2,4,5],[3,4,6],
  [3,5,6],[4,5,6]]
```

A triangulation in *mixed* lexicographical format could look like this:

```
manifold_2_6_1=[[1,2,3],[1,2,4],[1,3,5],[1,4,6],
  [1,5,6],[2,3,4],[3,4,5],[4,5,6]]
```

This format is **hard to parse**. Moreover, any *additional* information
about the triangulations, including information about homology groups or
orientability, for instance, requires additional files.

We thus decided to use a format that permits us to keep everything in
one place, including any additional attributes for a specific
triangulation. A desirable data format needs to satisfy the following
properties:

1. It should be easy to parse and modify, ideally in a number of
   programming languages.

2. It should be human-readable and `diff`-able in order to permit
   simplified comparisons.

3. It should scale reasonably well to larger triangulations.

After some considerations, we decided to opt for `gzip`-compressed JSON
files. [JSON](https://www.json.org) is well-specified and supported in
virtually all major programming languages out of the box. While the
compressed file is *not* human-readable on its own, the uncompressed
version can easily be used for additional data analysis tasks. This also
greatly simplifies maintenance operations on the dataset. While it can
be argued that there are formats that scale even better, they are
not well-applicable to our use case since each triangulation
typically consists of different numbers of top-level simplices. This
rules out column-based formats like [Parquet](https://parquet.apache.org/).

We are open to revisiting this decision in the future.

As for the *storage* of the data as such, we decided to keep only the
top-level simplices (as is done in the original format) since this
substantially saves disk space. The drawback is that the client has to
supply the remainder of the triangulation. Given that the triangulations
in our dataset are not too large, we deem this to be an acceptable
compromise. Moreover, data structures such as [simplex
trees](https://en.wikipedia.org/wiki/Simplex_tree) can be used to
further improve scalability if necessary.

The decision to keep only top-level simplices is **final**.

Finally, our data format includes, whenever possible and available,
additional information about a triangulation, including the [Betti
numbers](https://en.wikipedia.org/wiki/Betti_number) and a *name*,
i.e., a canonical description, of the topological space described
by the triangulation. We opted to minimize any inconvenience that
would arise from having to perform additional parsing operations.

## Acknowledgments

This work is dedicated to [Frank H. Lutz](https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/),
who passed away unexpectedly on November 10, 2023. May his memory be
a blessing.
