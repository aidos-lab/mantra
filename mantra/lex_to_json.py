"""Convert from lexicographical format to JSON.

The purpose of this script is to convert triangulations from
a lexicographical format to a JSON-based format. In the old,
i.e. lexicographical, format, triangulations can span *more*
than one line, and every line contains a different number of
vertices. This makes parsing the format cumbersome.

Storing the triangulations in JSON simplifies data handling,
storage, and downstream tasks.
"""

import argparse
import hashlib
import re
import sys

import numpy as np

from mantra.utils import store_triangulations


def process_triangulation(triangulation):
    """Process an individual triangulation.

    A triangulation is represented (following the original data format)
    as a newline-separated string of vertex indices. These indices will
    be parsed into a (nested) array of integers and returned in a dict,
    which contains additional information about the triangulation.

    Returns
    -------
    dict
        A dictionary containing information about the triangulation. The
        keys of dict are "triangulation", "dimension", and "n_vertices".
        The triangulation itself is stored as a nested list of vertices,
        representing the top-level simplices of the triangulation.
    """
    simplices = np.asarray(
        [
            np.fromstring(line, sep=",", dtype=int)
            for line in triangulation.split("\n")
        ]
    )

    dimensions = [len(simplex) - 1 for simplex in simplices]
    assert dimensions[1:] == dimensions[:-1]

    vertices = set(simplices.flatten())

    return {
        "triangulation": simplices.tolist(),
        "dimension": dimensions[0],
        "n_vertices": len(vertices),
    }


def parse_topological_type(s):
    """Parse the type field of a manifold.

    The type field of an entry consists of a flag indicating whether the
    manifold is orientable, followed by its genus. Optionally, there may
    be a canonical name, such as "Klein bottle."

    Parameters
    ----------
    s: string
        Input string containing topological type information.

    Returns
    -------
    dict
        A dictionary containing information about the topological type,
        using the keys "name" (for an optional canonical name, which is
        allowed to be the empty string), "orientable" (a boolean flag),
        and "genus" (an integer).
    """
    parts = s.split("=")
    assert len(parts) == 1 or len(parts) == 2

    # Every triangulation is supposed to have a name, even if it is an
    # empty one. The "orientable" attribute is only added when we know
    # the orientability, i.e. when it is either true or false.
    result = {"name": ""}

    # Case: There are two parts, delimited by "=". The second part is
    # the name of the object.
    if len(parts) == 2:
        result["name"] = parts[1].strip()

    # Case: There is only a single part, and the triangulation has no
    # canonical name. The second part of the condition prevents a set
    # of 3-manifolds, consisting of connected sums, from breaking the
    # parse process.
    if parts[0].strip().startswith("(") and "#" not in parts[0]:
        # Parse the "topological type" field, consisting of a bracketed
        # expression indicating whether the manifold is orientable, and
        # the genus.
        topological_type = parts[0]
        topological_type = topological_type.replace("(", "")
        topological_type = topological_type.replace(")", "")

        orientable, genus = topological_type.split(";")
        orientable = orientable.strip()

        result["genus"] = int(genus)

        assert orientable in ["+", "-"]

        if orientable == "+":
            result["orientable"] = True
        else:
            result["orientable"] = False

    # Case: There is only a single part, containing the canonical
    # name of the triangulation.
    else:
        result["name"] = parts[0].strip()

    return result


def parse_homology_groups(s):
    """Parse information about the homology groups of a manifold.

    Homology group information is represented as a tuple of ranks, with
    each entry either a number or a number with torsion information. An
    example would be "(1, 0 + Z_2, 0)".

    This information will be represented in two tuples:

    1. A "betti_numbers" list, consisting of a list of ranks of the
       homology groups, ignoring any torsion.
    2. A "torsion_coefficients" list, providing torsion coefficient
       information. Each torsion coefficient is stored as-is, i.e.,
       as a string. An empty torsion coefficient is indicated using
       an empty string.

    Parameters
    ----------
    s : string
        String containing information about the homology groups of the
        triangulation.

    Returns
    -------
    dict
        A dictionary containing the homology information of the
        triangulation (see above).
    """
    s = s.replace("(", "")
    s = s.replace(")", "")

    ranks = s.split(",")

    result = {
        "betti_numbers": [],
        "torsion_coefficients": [],
    }

    for rank in ranks:
        rank = rank.split("+")
        assert len(rank) == 1 or len(rank) == 2

        result["betti_numbers"].append(int(rank[0]))

        # Single entry only, so no torsion to consider.
        if len(rank) == 1:
            result["torsion_coefficients"].append("")

        # More than one entry. Store the torsion coefficient as-is.
        else:
            result["torsion_coefficients"].append(rank[1].strip())

    return result


def process_homology_groups_or_types(filename, parse_fn):
    """Process information about homology groups or topological types.

    The data format for the homology groups or the topological type of
    a triangulation is easier to parse than the triangulation since it
    only uses a single string, which provides the Betti numbers (and a
    summand for torsion), or a description of the type of the manifold
    that is triangulated.

    Parameters
    ----------
    parse_fn : callable
        Function to use for further processing each entry.

    Returns
    -------
    dict
        Dictionary, with keys indicating the respective triangulation
        and values being strings corresponding to homology groups, or
        type information, respectively. No further processing of each
        string is attempted.
    """
    with open(filename) as f:
        lines = f.readlines()

    matches = [re.match(r"(manifold_.*):\s+(.*)$", line) for line in lines]

    return {match.group(1): parse_fn(match.group(2)) for match in matches}


def hash_triangulation(triangulation):
    """Calculate a hash value (fingerprint) for a triangulation.

    Given a triangulation in the form of a list of top-level simplices,
    calculates a hash (fingerprint) that is order-invariant. This makes
    it possible to prevent duplicates when merging different datasets.

    Parameters
    ----------
    triangulation : list of list of int
        Triangulation, specified in the form of top-level simplices.

    Returns
    -------
    int
        Hash value, calculated based on SHA-256.
    """
    triangulation = [list(sorted(simplex)) for simplex in triangulation]
    triangulation = list(sorted(triangulation))

    return hashlib.sha256(str(triangulation).encode("utf-8")).hexdigest()


def find_duplicates(triangulations):
    """Find duplicate triangulations in a dict of triangulations.

    Given a dict of triangulations, uses `hash_triangulation()` to find
    duplicate triangulations.

    Parameters
    ----------
    triangulations : dict
        Dictionary of triangulations, with a "name" key (which is
        already guaranteed to be unique) but a possibly non-unique
        sequence of top-level simplices.

    Returns
    -------
    list
        List of names of triangulations that are not unique. The idea is
        that these names can be used to directly remove some keys from a
        dict. Since this function preserves insertion order, the *first*
        instance of each element will be preserved.
    """
    seen = set()
    names = []

    for name, data in triangulations.items():
        h = hash_triangulation(data["triangulation"])
        if h in seen:
            names.append(name)
        else:
            seen.add(h)

    return names


def process_triangulations(filename):
    """Process file in lexicographical format."""
    with open(filename) as f:
        lines = f.read()
        lines = lines.split("\n\n")

    # Get everything on a single line first and remove all empty lines
    # or blocks from the resulting array.
    lines = [line.replace("\n", "") for line in lines]
    lines = [line for line in lines if line]

    names = [re.match(r"(manifold_.*)=", line).group(1) for line in lines]

    lines = [re.sub(r"manifold_(.*)=", "", line) for line in lines]
    lines = [re.sub(r"\s+", "", line) for line in lines]
    lines = [re.sub(r"],", "]\n", line) for line in lines]
    lines = [re.sub(r"\]\]", "]", line) for line in lines]
    lines = [re.sub(r"\[\[", "[", line) for line in lines]
    lines = [re.sub(r"[\[\]]", "", line) for line in lines]

    assert len(lines) == len(names)

    triangulations = lines
    triangulations = {
        name: process_triangulation(triangulation)
        for name, triangulation in zip(names, triangulations)
    }

    return triangulations


def guess_name(triangulation):
    """Guess the name of a triangulation.

    This function makes use of the classification theorem for manifolds
    to guess the canonical name of a triangulation. Currently, this may
    only be taken seriously for 2-manifolds.

    Applying this to triangulations that already have a name will raise
    an error. This is deliberate.

    Parameters
    ----------
    triangulation : dict
        Triangulation with information about top-level simplices and all
        other attributes.

    Returns
    -------
    str
        The assigned name of this triangulation. Will be left empty in
        case no name can be found.
    """
    name = triangulation["name"]

    assert not name, RuntimeError("Triangulation already has a name")

    # Sorry, but we can only do this for 2-manifolds at the moment. Stay
    # tuned, folks!
    if triangulation["dimension"] != 2:
        return name

    if (
        "genus" not in triangulation
        or "betti_numbers" not in triangulation
        or "orientable" not in triangulation
    ):
        return name

    g = triangulation["genus"]
    betti_numbers = triangulation["betti_numbers"]

    assert g >= 2, f"Encountered unexpected genus 'g = {g}'"

    if triangulation["orientable"]:
        expected_betti_numbers = [1, 2 * g, 1]
        expected_torsion_coefficients = ["", "", ""]

        name = f"#^{g:d} T^2"
    else:
        expected_betti_numbers = [1, g - 1, 0]
        expected_torsion_coefficients = ["", "Z_2", ""]

        name = f"#^{g:d} RP^2"

    # We always check Betti numbers because every triangulation is
    # guaranteed to have it.
    assert (
        betti_numbers == expected_betti_numbers
    ), f"Encountered unexpected Betti numbers '{betti_numbers}'"

    # Only non-orientable triangulations need to have torsion
    # coefficients, but if they exist, we check them.
    assert (
        triangulation["orientable"] or "torsion_coefficients" in triangulation
    ), "Torsion coefficients are required for non-orientable manifolds"

    if "torsion_coefficients" in triangulation:
        torsion_coefficients = triangulation["torsion_coefficients"]

        assert (
            torsion_coefficients == expected_torsion_coefficients
        ), f"Encountered unexpected torsion coefficients '{torsion_coefficients}'"  # noqa

    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input file")
    parser.add_argument(
        "-H",
        "--homology",
        type=str,
        help="Homology information for triangulations",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="Type information for triangulations (optional)",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output file (optional)"
    )

    args = parser.parse_args()
    triangulations = process_triangulations(args.INPUT)

    if args.homology is not None:
        homology_groups = process_homology_groups_or_types(
            args.homology, parse_homology_groups
        )

        for manifold in triangulations:
            triangulations[manifold].update(homology_groups[manifold])

    if args.type is not None:
        types = process_homology_groups_or_types(
            args.type, parse_topological_type
        )

        for manifold in triangulations:
            triangulations[manifold].update(types[manifold])

    for _, triangulation in triangulations.items():
        if not triangulation["name"]:
            triangulation["name"] = guess_name(triangulation)

    print(
        f"Deduplicating {len(triangulations)} triangulations...",
        file=sys.stderr,
    )

    duplicates = find_duplicates(triangulations)
    for duplicate in duplicates:
        print(f"- {duplicate}", file=sys.stderr)
        triangulations.pop(duplicate)

    print(f"Storing {len(triangulations)} triangulations...", file=sys.stderr)

    # Turn ID into a separate attribute. This enables us to turn the
    # whole data set into a list of triangulations, making it easier
    # to add new triangulations later on.
    triangulations = [
        {"id": manifold, **triangulations[manifold]}
        for manifold in triangulations
    ]

    store_triangulations(triangulations, args.output)
