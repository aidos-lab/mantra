"""(Attempt) to create a random realization of a 2D triangulation.

This script is a work in progress and only tackles *orientable
2-manifolds* for the time being.
"""

import itertools
import json
import trimesh
import sys

import numpy as np

k = 32768
max_tries = 50000


def area(x, y, z):
    a = y - x
    b = z - x

    cross = np.cross(a, b)
    return float(0.5 * np.linalg.norm(cross))


def areas(top_level_simplices, coordinates):
    result = []

    for triangle in top_level_simplices:
        x = coordinates[triangle[0] - 1]
        y = coordinates[triangle[1] - 1]
        z = coordinates[triangle[2] - 1]

        A = area(x, y, z)
        result.append(A)

    return result


# TODO: Check if this actually works, I cobbled it together from
# multiple sources.
def intersects(start, end, triangle):
    v0, v1, v2 = triangle
    d = end - start

    # Compute the triangle's normal vector using cross product of two
    # edges of the triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)

    denominator = np.dot(normal, d)
    if abs(denominator) < 1e-10:
        return False

    t = np.dot(normal, v0 - start) / denominator
    if t < 0 or t > 1:
        return False

    intersection_point = start + t * d

    # Check if the intersection point is inside the triangle using
    # barycentric coordinates
    v0_to_ip = intersection_point - v0
    dot00 = np.dot(edge1, edge1)
    dot01 = np.dot(edge1, edge2)
    dot02 = np.dot(edge1, v0_to_ip)
    dot11 = np.dot(edge2, edge2)
    dot12 = np.dot(edge2, v0_to_ip)

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) and (v >= 0) and (u + v <= 1)


def realize_triangulation(data):
    assert data["dimension"] == 2, RuntimeError("Unexpected dimension")

    n = data["n_vertices"]
    top_level_simplices = data["triangulation"]
    simplices = set([tuple(s) for s in top_level_simplices])
    max_dim = len(next(iter(simplices)))

    for simplex in top_level_simplices:
        for dim in range(1, max_dim):
            simplices.update(s for s in itertools.combinations(simplex, r=dim))

    # Get every complex into lexicographic order. This
    # requires converting everything back to a list.
    simplices = list(simplices)
    simplices.sort()
    simplices.sort(key=len)

    # Store triangulation (first argument) and its area (second
    # argument).
    best_triangulation = (None, None)

    # Let's try this only a couple of times...
    for i in range(max_tries):
        X = rng.integers(low=1, high=k + 1, size=(n, 3))

        triangles = [simplex for simplex in simplices if len(simplex) == 3]
        edges = [simplex for simplex in simplices if len(simplex) == 2]

        invalid = False

        for triangle in triangles:
            for edge in edges:
                u, v = edge
                # Only use combinatorially distinct edges since it is
                # sufficient to test them for intersection.
                if u in triangle or v in triangle:
                    continue

                end = X[u - 1]
                start = X[v - 1]
                coordinates = [X[u - 1] for u in triangle]

                if intersects(start, end, coordinates):
                    invalid = True
                    break

            # Invalid realization, try again!
            if invalid:
                break

        if not invalid:
            # print("Required", i, "tries:", X / k)

            all_areas = areas(top_level_simplices, X / k)
            area_difference = np.max(all_areas) - np.min(all_areas)

            print(area_difference)

            if (
                best_triangulation[0] is None
                or area_difference < best_triangulation[1]
            ):
                best_triangulation = (X / k, area_difference)

            # plot(data["id"], data["name"], top_level_simplices, X)
            # break


def plot(id, name, top_level_simplices, coordinates):
    coordinates = coordinates.astype(float)
    coordinates /= k

    faces = np.asarray(top_level_simplices)
    faces = faces - 1

    mesh = trimesh.Trimesh(vertices=coordinates, faces=faces)
    mesh.fix_normals()

    points = trimesh.points.PointCloud(coordinates, colors=[0, 0, 0, 255])

    scene = mesh.scene()
    scene.add_geometry(points)

    scene.show(smooth=False)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        triangulations = json.load(f)

    rng = np.random.default_rng(42)

    for triangulation in triangulations:
        if (
            triangulation["name"] == "S^2"
            and triangulation["n_vertices"] == 10
        ):
            realize_triangulation(triangulation)
            break
