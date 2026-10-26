import pytest
import torch
from torch_geometric.data import Data

from mantra.transforms import EulerCharacteristicTransform

TRIANGLE = [[1, 2, 3]]
COORDS = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def along_x(transform):
    """Transform the triangle along the x-axis as the only direction."""
    transform._directions = torch.tensor([[1.0], [0.0]])
    return transform(Data(coords=COORDS, triangulation=TRIANGLE)).ect.long()


def test_shape_dtype_and_names():
    transform = EulerCharacteristicTransform(
        num_directions=5, resolution=7, source="pos", target="curves"
    )
    data = transform(Data(pos=COORDS, triangulation=TRIANGLE))
    assert "ect" not in data
    assert data.curves.shape == (1, 5, 7)
    assert data.curves.dtype == torch.int16


# Vertex heights along x are 0, 1, 0; thresholds span [-radius, radius].
@pytest.mark.parametrize(
    "max_rank, radius, expected",
    [
        (None, 2.0, [0, 0, 1, 1, 1]),  # filled triangle: contractible
        (1, 2.0, [0, 0, 1, 0, 0]),  # one-skeleton closes into a circle
        (0, 2.0, [0, 0, 2, 3, 3]),  # vertex counts
        (0, 0.5, [0, 2, 2]),  # faces above the radius are not counted
    ],
)
def test_curves(max_rank, radius, expected):
    transform = EulerCharacteristicTransform(
        num_directions=1,
        resolution=len(expected),
        radius=radius,
        max_rank=max_rank,
    )
    assert along_x(transform)[0, 0].tolist() == expected


def test_per_rank_keeps_the_face_counts_apart():
    transform = EulerCharacteristicTransform(
        num_directions=1, resolution=5, radius=2.0, max_rank=1, per_rank=True
    )
    ect = along_x(transform)
    assert ect.shape == (1, 2, 1, 5)
    assert ect[0, :, 0].tolist() == [[0, 0, 2, 3, 3], [0, 0, 1, 3, 3]]
    assert (ect[0, 0] - ect[0, 1]).tolist() == [[0, 0, 1, 0, 0]]


def test_directions_depend_only_on_seed():
    a = EulerCharacteristicTransform(num_directions=8, seed=3)
    b = EulerCharacteristicTransform(num_directions=8, seed=3)
    c = EulerCharacteristicTransform(num_directions=8, seed=4)
    assert torch.equal(a.directions(4), b.directions(4))
    assert not torch.equal(a.directions(4), c.directions(4))
    assert torch.allclose(a.directions(4).norm(dim=0), torch.ones(8))

