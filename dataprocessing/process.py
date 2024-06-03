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

import argparse
import json
import re
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import pydantic


import os.path as osp
import ssl
import sys
import urllib
from typing import Optional

from torch_geometric.data.makedirs import makedirs


class Triangulation(pydantic.BaseModel):
    id: str
    triangulation: List[List[int]]
    dimension: int
    n_vertices: int

    @pydantic.model_validator(mode="after")
    def check_model(self):
        # Check if triangulations have the same length of 3.
        if not all([len(item) == 3 for item in self.triangulation]):
            raise pydantic.ValidationError(
                "Every element in the list should have length 3."
            )

        # Triangulation has 1 based indexing, so max is number of vertices.
        if not np.array(self.triangulation).max() == self.n_vertices:
            raise ValueError(
                f"Number of vertices in the triangulation ({np.array(self.triangulation).max()}) does not coincide with n_vertices ({self.n_vertices})"
            )

        # Check if dimension is correct.
        assert self.dimension == len(self.triangulation[0]) - 1


class TopologicalType(pydantic.BaseModel):
    id: str
    orientable: bool
    genus: int
    name: Optional[str]


class Homology(pydantic.BaseModel):
    id: str
    torsion_coefficients: List[str]
    betti_numbers: List[int]

    @pydantic.model_validator(mode="after")
    def check_model(self):
        assert len(self.betti_numbers) == 3
        assert len(self.torsion_coefficients) == 3


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

    # json.loads loads string into List[List[int]].
    return Triangulation(
        id=line[0],
        dimension=match.group(1),
        n_vertices=match.group(2),
        triangulation=json.loads(line[1]),
    )


def process_triangulation(content: str) -> List[Triangulation]:
    lines = content.removesuffix("\n\n")
    lines = re.split("\n\n", lines)
    lines = [re.sub("\n\s+", "", line) for line in lines]

    return [process_triangulation_line(line) for line in lines]


def process_type_line(line: str) -> TopologicalType:
    """Parses a single line with the homology type.

    The line has to have the format
    'manifold_{dim}_{n_vert}_{non-unique-id}:  ( {orientable} ; {genus} ) =
    {name}'. The name field is optional and may be omitted. The function returns
    a dictionary with all fields parsed.

    Examples:
    - 'manifold_2_9_645:  ( - ; 1 ) = RP^2'
    - 'manifold_2_9_646:  ( - ; 2 ) = Klein bottle'
    - 'manifold_2_9_587:  ( - ; 3 )'

    Parameters
    ----------
    line : string
        A single line to parse in the format above.

    Returns:
        Dictionary containing the manifold id, the corresponding homology groups,
        orientation and if present the name.
    """
    match = re.match(r"(manifold_.*):\s+\( ([+-])\s;\s(\d)\s\)(\s=\s)?(.*)?", line)
    return TopologicalType(
        id=match.group(1),
        orientable=True if match.group(2) == "+" else False,
        genus=int(match.group(3)),
        name=match.group(5),
    )


def process_type(content: str) -> List[TopologicalType]:
    lines = content.removesuffix("\n").split("\n")
    return [process_type_line(line) for line in lines]


def process_homology_line(line: str) -> Homology:
    match = re.match(r"(manifold_.*):\s+\((.*)\)", line)
    tc, bn = [], []
    for rank in match.group(2).split(", "):
        rank_match = re.match("(\d+)(\s\+\s)?(.*)?", rank)
        bn.append(int(rank_match.group(1)))
        tc.append(rank_match.group(3))

    return Homology(id=match.group(1), torsion_coefficients=tc, betti_numbers=bn)


def process_homology(lines: str) -> List[Homology]:
    lines = lines.removesuffix("\n").split("\n")
    return [process_homology_line(line) for line in lines]


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
    filename_homology: str | None = None,
    filename_type: str | None = None,
) -> List[Dict]:
    homology_groups, types = {}, {}

    # Parse triangulations
    with open(filename_triangulation) as f:
        lines = f.read()
    triangulations = process_triangulation(lines)

    # Parse homology
    if filename_homology:
        with open(filename_homology) as f:
            lines = f.read()
        homology_groups = process_homology(lines)

    # Parse type
    if filename_type:
        with open(filename_type) as f:
            lines = f.read()
        types = process_type(lines)

    if filename_homology or filename_type:
        triangulations = merge_triangulation(triangulations, homology_groups, types)

    return triangulations


def download_url(
    url: str, folder: str, log: bool = True, filename: Optional[str] = None
):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log and "pytest" not in sys.modules:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log and "pytest" not in sys.modules:
        print(f"Downloading {url}", file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def download_files() -> None:
    """
    Download the files from the website.
    """
    file_names = [
        "2_manifolds_all.txt",
        "2_manifolds_all_type.txt",
        "2_manifolds_all_hom.txt",
    ]
    url = "https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/"
    for file_name in file_names:
        download_url(url + file_name, "./data")


def main():
    download_files()
    triangulations = process_manifolds(
        "2_manifolds_all.txt",
        "2_manifolds_all_hom.txt",
        "2_manifolds_all_type.txt",
    )
    with open("triangulations.json") as f:
        json.dump(triangulations, f)


if __name__ == "__main__":
    main()
