"""Convert from lexicographical format to line-based format.

The purpose of this script is to convert triangulations from
a lexicographical format to a line-based format. In the old,
i.e. lexicographical, format, triangulations can span *more*
than one line, and every line contains a different number of
vertices. This makes parsing the format cumbersome.

As an alternative, this script parses the triangulations and
represents them as JSON objects. This facilitates *storing*,
as well as *processing* the data types.
"""

import json
import re
from typing import List, Literal, Optional, Dict
import numpy as np
import pandas as pd
import pydantic
from torch_geometric.data import extract_gz, makedirs, download_url


# Constants

URL = "https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/"


################################################################################
### Pydantic helper classes
################################################################################


class Triangulation(pydantic.BaseModel):
    id: str
    triangulation: List[List[int]]
    dimension: int
    n_vertices: int

    @pydantic.model_validator(mode="after")
    def check_model(self):
        # Triangulation has 1 based indexing, so max is number of vertices.
        if not np.array(self.triangulation).max() == self.n_vertices:
            raise ValueError(
                f"Number of vertices in the triangulation ({np.array(self.triangulation).max()}) does not coincide with n_vertices ({self.n_vertices})"
            )

        # Check if dimension is correct.
        assert self.dimension == len(self.triangulation[0]) - 1


class TopologicalType(pydantic.BaseModel):
    id: str
    name: Optional[str]


class Homology(pydantic.BaseModel):
    id: str
    torsion_coefficients: List[str]
    betti_numbers: List[int]


################################################################################
### Processing scripts
################################################################################


def process_triangulation_line(line: str) -> Triangulation:
    """Parses a single line of the triangluation.

    A triangulation is represented (following the original data format)
    as a newline-separated string of vertex indices. These indices will
    be parsed into a (nested) array of integers and returned.

    Example:
    'manifold_{dim}_{n_vert}_{non-unique-id}=[[1,2,3],...,[1,4,3]]'

    Returns
    -------
    dict
        Dictionary, with keys indicating the respective triangulation
        and values being strings corresponding to homology groups, or
        type information, respectively. No further processing of each
        string is attempted.
    """
    line = re.split("=", line)

    # Parse dimension and n_vertices from id.
    match = re.match("manifold_(\d+)_(\d+)", line[0])
    if match is None:
        print(line)
        raise "hello"

    # json.loads loads string into List[List[int]].
    return Triangulation(
        id=line[0],
        dimension=match.group(1),
        n_vertices=match.group(2),
        triangulation=json.loads(line[1]),
    )


def process_triangulation(content: str) -> List[Triangulation]:
    lines = content.removesuffix("\n\n")
    lines = lines.removeprefix("\n")
    lines = re.split("\n\n", lines)
    lines = [re.sub("\n\s+", "", line) for line in lines]

    return [process_triangulation_line(line) for line in lines]


def process_3_manifold_type_line(line: str) -> TopologicalType:
    match = re.match(r"(manifold_.*):(.*)?", line)
    return TopologicalType(
        id=match.group(1),
        name=match.group(2),
    )


def process_2_manifold_type_line(line: str) -> TopologicalType:
    match = re.match(r"(manifold_.*):\s+\( ([+-])\s;\s(\d)\s\)(\s=\s)?(.*)?", line)
    return TopologicalType(
        id=match.group(1),
        name=match.group(2),
    )


def process_2_manifold_type(content: str) -> List[TopologicalType]:
    lines = content.removesuffix("\n").split("\n")
    return [process_2_manifold_type_line(line) for line in lines]


def process_3_manifold_type(content: str) -> List[TopologicalType]:
    lines = content.removesuffix("\n").split("\n")
    return [process_3_manifold_type_line(line) for line in lines]


def process_2_manifold_homology_line(line: str) -> Homology:
    match = re.match(r"(manifold_.*):\s+\((.*)\)", line)
    tc, bn = [], []
    for rank in match.group(2).split(", "):
        rank_match = re.match("(\d+)(\s\+\s)?(.*)?", rank)
        bn.append(int(rank_match.group(1)))
        tc.append(rank_match.group(3))

    return Homology(id=match.group(1), torsion_coefficients=tc, betti_numbers=bn)


def process_2_manifold_homology(lines: str) -> List[Homology]:
    lines = lines.removesuffix("\n").split("\n")
    return [process_2_manifold_homology_line(line) for line in lines]


def merge_triangulation(
    triangulation: List[Triangulation],
    homology_groups: List[Homology],
    types: List[TopologicalType],
) -> List[Triangulation]:
    df_triangulation = pd.DataFrame.from_records(
        [tr.__dict__ for tr in triangulation]
    ).set_index("id")

    if homology_groups:
        df_homology = pd.DataFrame.from_records(
            [h.__dict__ for h in homology_groups]
        ).set_index("id")
        df_triangulation = df_triangulation.join(df_homology, "id")

    if types:
        df_types = pd.DataFrame.from_records([t.__dict__ for t in types]).set_index(
            "id"
        )
        df_triangulation = df_triangulation.join(df_types, "id")
    return df_triangulation.to_dict(orient="records")


def process_manifolds(
    filename_triangulation: str,
    filename_homology: str,
    filename_type: str,
    manifold_dim: Literal[2, 3] = 2,
) -> List[Dict]:
    homology_groups, types = {}, {}

    # Parse triangulations
    with open(filename_triangulation) as f:
        lines = f.read()
    triangulations = process_triangulation(lines)

    # Parse homology
    with open(filename_homology) as f:
        lines = f.read()
    homology_groups = process_2_manifold_homology(lines)

    # Parse type for 2 manifolds
    if manifold_dim == 2:
        with open(filename_type) as f:
            lines = f.read()
        types = process_2_manifold_type(lines)
    # Parse type for 3 manifolds
    elif manifold_dim == 3:
        with open(filename_type) as f:
            lines = f.read()
        types = process_3_manifold_type(lines)

    if filename_homology or filename_type:
        triangulations = merge_triangulation(triangulations, homology_groups, types)

    return triangulations


def download_files(file_names) -> None:
    """
    Download the files from the website.
    """
    for file_name in file_names:
        download_url(URL + file_name, "./data")


def main(manifold_dim):
    """
    Processes the 2 and 3 dimensional manifolds.
    """
    if manifold_dim == 2:
        file_names = [
            f"{manifold_dim}_manifolds_all.txt",
            f"{manifold_dim}_manifolds_all_hom.txt",
            f"{manifold_dim}_manifolds_all_type.txt",
            f"{manifold_dim}_manifolds_10_all.txt",
            f"{manifold_dim}_manifolds_10_all_hom.txt",
            f"{manifold_dim}_manifolds_10_all_type.txt",
        ]
    elif manifold_dim == 3:
        file_names = [
            f"{manifold_dim}_manifolds_all.txt",
            f"{manifold_dim}_manifolds_all_hom.txt",
            f"{manifold_dim}_manifolds_all_type.txt",
            f"{manifold_dim}_manifolds_10_all.txt.gz",
            f"{manifold_dim}_manifolds_10_all_hom.txt",
            f"{manifold_dim}_manifolds_10_all_type.txt",
        ]

    makedirs("./data")
    download_files(file_names=file_names)
    gz_files = [file for file in file_names if file.endswith("gz")]
    if gz_files:
        for gz_file in gz_files:
            extract_gz("./data/" + gz_file, "./data")

    triangulations = process_manifolds(
        f"./data/{manifold_dim}_manifolds_all.txt",
        f"./data/{manifold_dim}_manifolds_all_hom.txt",
        f"./data/{manifold_dim}_manifolds_all_type.txt",
        manifold_dim=manifold_dim,
    )

    triangulations_10 = process_manifolds(
        f"./data/{manifold_dim}_manifolds_10_all.txt",
        f"./data/{manifold_dim}_manifolds_10_all_hom.txt",
        f"./data/{manifold_dim}_manifolds_10_all_type.txt",
        manifold_dim=manifold_dim,
    )

    # Merge the results
    triangulations += triangulations_10

    with open(f"./processed/{manifold_dim}_manifolds.json", "w") as f:
        json.dump(triangulations, f)


if __name__ == "__main__":
    makedirs("./data")
    makedirs("./processed")
    main(manifold_dim=2)
    main(manifold_dim=3)
