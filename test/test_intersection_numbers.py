import pytest
import torch
from torch_geometric.data import Data

from mantra.representations import IncidenceSimplicialComplex
from mantra.transforms import TriangleIntersectionNumbers

# Cone over a square boundary; its triangles in lexicographic order are
# (0,1,2), (0,1,4), (0,2,3), (0,3,4).
TRIANGULATION = [[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 2]]
NUMBERS = [[3, 0, 2, 5], [4, 1, 0, -2], [1, 1, 2, 9], [0, 0, 0, 7]]


@pytest.mark.parametrize("numbers", [NUMBERS, torch.tensor(NUMBERS)])
def test_values_follow_the_lexicographic_triangle_order(numbers):
    data = TriangleIntersectionNumbers()(
        Data(triangulation=TRIANGULATION, intersection_numbers=numbers)
    )
    assert torch.equal(data.kappa_2, torch.tensor([0.0, -2.0, 5.0, 0.0]))
    assert torch.equal(
        data.simplex_index_2,
        torch.tensor([[0, 0, 0, 0], [1, 1, 2, 3], [2, 4, 3, 4]]),
    )


def test_order_matches_the_rank_2_skeleton_of_the_representations():
    data = Data(
        triangulation=[[1, 2, 3, 4], [1, 2, 3, 5]],
        intersection_numbers=torch.zeros(0),
        dimension=3,
    )
    incidence = IncidenceSimplicialComplex(signed=True)
    data = incidence(TriangleIntersectionNumbers()(data))
    skeleton = incidence.skeleton(data.simplex_trie, 2)
    assert (data.simplex_index_2.t() + 1).tolist() == [
        list(s) for s in skeleton
    ]
    assert torch.equal(data.kappa_2, torch.zeros(7))


def test_entries_outside_the_triangulation_or_malformed_raise():
    with pytest.raises(ValueError, match="outside"):
        TriangleIntersectionNumbers()(
            Data(
                triangulation=TRIANGULATION,
                intersection_numbers=[[1, 2, 3, 1]],
            )
        )
    with pytest.raises(RuntimeError):
        TriangleIntersectionNumbers()(
            Data(triangulation=TRIANGULATION, intersection_numbers=[[0, 1, 2]])
        )
