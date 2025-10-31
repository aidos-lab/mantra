"""Utility functions for interacting with triangulations."""

import json
import re
import sys

from contextlib import nullcontext


def store_triangulations(triangulations, output=None):
    """Store triangulations in "pretty" format.

    This function stores a list of triangulations in a somewhat
    "prettified" JSON format (one line per simplex). The output
    may either be `stdout` or a file.

    Parameters
    ----------
    triangulations : list of dict
        List of triangulations to store.

    output : str or None
        Output file. If `None`, will write the list to `stdout`.
    """
    with (
        open(output, "w") if output is not None else nullcontext(sys.stdout)
    ) as f:
        result = json.dumps(triangulations, indent=2)

        regex = re.compile(
            r"^(\s+)\[(.*?)\]([,]\s+?)", re.MULTILINE | re.DOTALL
        )

        def prettify_triangulation(match):
            """Auxiliary function for pretty-printing a triangulation.

            Given a match that contains *all* the top-level vertices
            involved in the triangulation, this function will ensure
            that they are all printed on individual lines. Plus, any
            indent is preserved.
            """
            groups = match.groups()
            indent = match.group(1)
            vertex = match.group(2)
            vertex = vertex.replace("\n", "")
            vertex = re.sub(r"\s+", "", vertex)

            result = f"{indent}[{vertex}]"

            if len(groups) == 3:
                result += ",\n"

            return result

        result = regex.sub(prettify_triangulation, result)

        # Fix indent of "triangulation" fields afterwards. This ensures
        # that the closing bracket of the triangulation key aligns with
        # the start.
        regex = re.compile(
            r"^(\s+)\"triangulation\":.*?\]\]", re.MULTILINE | re.DOTALL
        )

        indents = [len(match.group(1)) for match in regex.finditer(result)]

        assert len(indents) != 0
        assert indents[0] > 0
        assert sum(indents) / indents[0] == len(indents)

        indent = " " * indents[0]
        result = result.replace("]],", f"]\n{indent}],")

        f.write(result)
