"""Set property for a list of triangulations.

The purpose of this script is to assign a property to a list of
triangulations. The script operates based on five inputs:

    1. A file (in lexicographical format) for identifying all relevant
       triangulations.

    2. The name of the property to set.

    3. The value the property should take on all triangulations
       specified in the input file.

    4. The value the property should take for all other triangulations.

    5. The file that contains all triangulations to which the property
       should be added. Note that this is different from the file that
       *identifies* all triangulations.
"""

import argparse
import json
import re
import sys

from mantra.utils import store_triangulations


def maybe_coerce(s):
    """Try to coerce a string to a JSON data type if possible."""
    s = s.strip()

    try:
        return json.loads(s)
    except:  # noqa
        pass

    return s


def get_identifiers(filename):
    """Return identifiers from a file in lexicographical format."""
    with open(filename) as f:
        lines = f.read()
        lines = lines.split("\n\n")

    # Get everything on a single line first and remove all empty lines
    # or blocks from the resulting array.
    lines = [line.replace("\n", "") for line in lines]
    lines = [line for line in lines if line]

    identifiers = [
        re.match(r"(manifold_.*)=", line).group(1) for line in lines
    ]

    return set(identifiers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input file (in JSON format)")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Name of the property to set",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--id-file",
        type=str,
        help="File containing all triangulations to apply the property to. "
        "This file must be in lexicographical format.",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--value",
        type=str,
        help="Value to set for all identified triangulations",
        required=True,
    )
    parser.add_argument(
        "-O",
        "--other-value",
        type=str,
        help="Value to set for all other triangulations",
        required=True,
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output file (optional)"
    )

    args = parser.parse_args()

    identifiers = get_identifiers(args.id_file)
    key = args.name

    value = maybe_coerce(args.value)
    other_value = maybe_coerce(args.other_value)

    with open(args.INPUT) as f:
        triangulations = json.load(f)

    for triangulation in triangulations:
        # Do not overwrite stuff but rather raise an error; this
        # should *not* happen.
        assert key not in triangulation.keys(), "Property name must not exist"

        if triangulation["id"] in identifiers:
            triangulation[key] = value
            identifiers.remove(triangulation["id"])
        else:
            triangulation[key] = other_value

    if identifiers:
        print("Did not find the following triangulations:")

        for identifier in identifiers:
            print(f"- {identifier}", file=sys.stderr)

    store_triangulations(triangulations, args.output)
