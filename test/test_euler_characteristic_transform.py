import torch
from torch_geometric.data import Data

from mantra.transforms import EulerCharacteristicTransform

TRIANGLE = [[1, 2, 3]]
COORDS = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def ect_along(transform, coords, triangulation, direction):
    """ECT curve of one explicit direction."""
    transform._directions = torch.tensor(direction).float().reshape(-1, 1)
    data = transform(Data(coords=coords, triangulation=triangulation))
    return data.ect[0, 0].long()


def test_output_shape_and_dtype():
    transform = EulerCharacteristicTransform(num_directions=5, resolution=7)
    data = transform(Data(coords=COORDS, triangulation=TRIANGLE))
    assert data.ect.shape == (1, 5, 7)
    assert data.ect.dtype == torch.int16


def test_filled_triangle_is_contractible():
    transform = EulerCharacteristicTransform(
        num_directions=1, resolution=5, radius=2.0
    )
    # Thresholds -2, -1, 0, 1, 2; vertex heights along x are 0, 1, 0.
    curve = ect_along(transform, COORDS, TRIANGLE, [1.0, 0.0])
    assert curve.tolist() == [0, 0, 1, 1, 1]


def test_one_skeleton_closes_into_a_circle():
    transform = EulerCharacteristicTransform(
        num_directions=1, resolution=5, radius=2.0, max_rank=1
    )
    # Two vertices and one edge below t = 0, all three vertices and
    # three edges from t = 1 on: the boundary of the triangle has
    # Euler characteristic 0.
    curve = ect_along(transform, COORDS, TRIANGLE, [1.0, 0.0])
    assert curve.tolist() == [0, 0, 1, 0, 0]


def test_vertices_only():
    transform = EulerCharacteristicTransform(
        num_directions=1, resolution=5, radius=2.0, max_rank=0
    )
    curve = ect_along(transform, COORDS, TRIANGLE, [1.0, 0.0])
    assert curve.tolist() == [0, 0, 2, 3, 3]


def test_faces_above_radius_are_not_counted():
    transform = EulerCharacteristicTransform(
        num_directions=1, resolution=3, radius=0.5, max_rank=0
    )
    curve = ect_along(transform, COORDS, TRIANGLE, [1.0, 0.0])
    assert curve.tolist() == [0, 2, 2]


def test_directions_depend_only_on_seed():
    a = EulerCharacteristicTransform(num_directions=8, seed=3)
    b = EulerCharacteristicTransform(num_directions=8, seed=3)
    c = EulerCharacteristicTransform(num_directions=8, seed=4)
    assert torch.equal(a.directions(4), b.directions(4))
    assert not torch.equal(a.directions(4), c.directions(4))
    assert torch.allclose(a.directions(4).norm(dim=0), torch.ones(8))


def test_respects_source_and_target_names():
    transform = EulerCharacteristicTransform(
        num_directions=2, resolution=3, source="pos", target="curves"
    )
    data = transform(Data(pos=COORDS, triangulation=TRIANGLE))
    assert "ect" not in data
    assert data.curves.shape == (1, 2, 3)


def test_per_rank_keeps_the_face_counts_apart():
    transform = EulerCharacteristicTransform(
        num_directions=1, resolution=5, radius=2.0, max_rank=1, per_rank=True
    )
    transform._directions = torch.tensor([[1.0], [0.0]])
    data = transform(Data(coords=COORDS, triangulation=TRIANGLE))
    assert data.ect.shape == (1, 2, 1, 5)
    assert data.ect[0, 0, 0].tolist() == [0, 0, 2, 3, 3]
    assert data.ect[0, 1, 0].tolist() == [0, 0, 1, 3, 3]
    ect = EulerCharacteristicTransform(
        num_directions=1, resolution=5, radius=2.0, max_rank=1
    )
    assert torch.equal(
        (data.ect[0, 0] - data.ect[0, 1]).long(),
        ect_along(ect, COORDS, TRIANGLE, [1.0, 0.0]).unsqueeze(0),
    )
