import pytest
import torch
from torch_geometric.data import Data

from mantra.representations import IncidenceSimplicialComplex
from mantra.transforms import (
    CandidateTriangleIntersectionNumbers,
    TriangleIntersectionNumbers,
)

# Cone over the boundary of a tetrahedron (vertex 1 is the origin). Its
# ten triangles in lexicographic order are the six through the origin,
# then the boundary triangles (1,2,3), (1,2,4), (1,3,4), (2,3,4).
TRIANGULATION = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5]]
# Two boundary entries, one self-intersection, and two rows through the
# origin (the canonical class), which no triangle target reads.
NUMBERS = [
    [3, 1, 2, 5],
    [4, 2, 1, -2],
    [2, 2, 1, 9],
    [0, 1, 2, 7],
    [0, 0, 0, -125],
]


@pytest.mark.parametrize("numbers", [NUMBERS, torch.tensor(NUMBERS)])
def test_boundary_triangles_in_lexicographic_order(numbers):
    data = TriangleIntersectionNumbers()(
        Data(triangulation=TRIANGULATION, intersection_numbers=numbers)
    )
    assert torch.equal(data.kappa_2, torch.tensor([5.0, -2.0, 0.0, 0.0]))
    assert torch.equal(
        data.simplex_index_2,
        torch.tensor([[1, 1, 1, 2], [2, 2, 3, 3], [3, 4, 4, 4]]),
    )
    assert data.triangle_mask.tolist() == [False] * 6 + [True] * 4


def test_mask_selects_the_rank_2_skeleton_of_the_representations():
    data = Data(
        triangulation=[[1, 2, 3, 4], [1, 2, 3, 5]],
        intersection_numbers=torch.zeros(0),
        dimension=3,
    )
    incidence = IncidenceSimplicialComplex(signed=True)
    data = incidence(TriangleIntersectionNumbers()(data))
    skeleton = incidence.skeleton(data.simplex_trie, 2)
    assert data.triangle_mask.tolist() == [False] * 5 + [True] * 2
    assert (data.simplex_index_2.t() + 1).tolist() == [
        list(s) for s, keep in zip(skeleton, data.triangle_mask) if keep
    ]
    assert torch.equal(data.kappa_2, torch.zeros(2))


# Cone over the boundary of an octahedron (vertex 1 the origin, 2/3,
# 4/5, 6/7 the opposite pairs): (2,3,4) is in range but not a face.
OCTAHEDRON = [[1, a, b, c] for a in (2, 3) for b in (4, 5) for c in (6, 7)]


def test_entries_outside_the_triangulation_or_malformed_raise():
    with pytest.raises(ValueError, match="outside"):
        TriangleIntersectionNumbers()(
            Data(triangulation=OCTAHEDRON, intersection_numbers=[[1, 2, 3, 1]])
        )
    with pytest.raises(ValueError, match="outside"):
        TriangleIntersectionNumbers()(
            Data(
                triangulation=TRIANGULATION,
                intersection_numbers=[[1, 2, 5, 1]],
            )
        )
    with pytest.raises(RuntimeError):
        TriangleIntersectionNumbers()(
            Data(triangulation=TRIANGULATION, intersection_numbers=[[1, 2, 3]])
        )
    with pytest.raises(ValueError, match="missing"):
        TriangleIntersectionNumbers()(Data(triangulation=TRIANGULATION))
    with pytest.raises(ValueError, match="origin"):
        TriangleIntersectionNumbers()(
            Data(
                triangulation=TRIANGULATION,
                intersection_numbers=NUMBERS,
                coords=torch.ones(5, 3),
            )
        )


# The four boundary points of TRIANGULATION as one 2-face, in rows
# (face, point): its four
# triples are all triangles of the full cone, and two of the half cone.
FACE = [[0, 1], [0, 2], [0, 3], [0, 4]]
# Two triangular 2-faces, (1,2,3) and (2,3,4).
TWO_TRIANGLES = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4]]
HALF_CONE = [[1, 2, 3, 4], [1, 2, 3, 5]]


def test_candidates_of_a_face_with_present_and_absent_triangles():
    full = CandidateTriangleIntersectionNumbers()(
        Data(
            triangulation=TRIANGULATION,
            intersection_numbers=NUMBERS,
            face2_points=FACE,
        )
    )
    assert torch.equal(
        full.kappa_candidate, torch.tensor([5.0, -2.0, 0.0, 0.0])
    )
    assert torch.equal(
        full.candidate_index_2,
        torch.tensor([[1, 1, 1, 2], [2, 2, 3, 3], [3, 4, 4, 4]]),
    )
    assert full.candidate_present.tolist() == [True] * 4

    half = CandidateTriangleIntersectionNumbers()(
        Data(
            triangulation=HALF_CONE,
            intersection_numbers=NUMBERS[:2],
            face2_points=FACE,
        )
    )
    assert torch.equal(
        half.kappa_candidate, torch.tensor([5.0, -2.0, 0.0, 0.0])
    )
    assert half.candidate_present.tolist() == [True, True, False, False]
    assert torch.equal(half.candidate_index_2, full.candidate_index_2)


def test_small_faces_and_collinear_triples_give_no_candidates():
    small = CandidateTriangleIntersectionNumbers()(
        Data(
            triangulation=TRIANGULATION,
            intersection_numbers=NUMBERS,
            face2_points=TWO_TRIANGLES,
        )
    )
    assert small.kappa_candidate.shape == (0,)
    assert small.candidate_index_2.shape == (3, 0)
    kept = CandidateTriangleIntersectionNumbers(min_points=3)(
        Data(
            triangulation=TRIANGULATION,
            intersection_numbers=NUMBERS,
            face2_points=TWO_TRIANGLES,
        )
    )
    assert torch.equal(kept.kappa_candidate, torch.tensor([5.0, 0.0]))

    # Points 1, 2, 3 on a line: of the four triples only those with 4 remain.
    coords = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0]]
    data = CandidateTriangleIntersectionNumbers()(
        Data(
            triangulation=TRIANGULATION,
            intersection_numbers=NUMBERS,
            face2_points=FACE,
            coords=coords,
        )
    )
    assert data.candidate_index_2.t().tolist() == [
        [1, 2, 4],
        [1, 3, 4],
        [2, 3, 4],
    ]


def test_candidates_need_the_faces_and_the_origin_first():
    with pytest.raises(ValueError, match="face2_points"):
        CandidateTriangleIntersectionNumbers()(
            Data(triangulation=TRIANGULATION, intersection_numbers=NUMBERS)
        )
    with pytest.raises(ValueError, match="origin"):
        CandidateTriangleIntersectionNumbers()(
            Data(
                triangulation=TRIANGULATION,
                intersection_numbers=NUMBERS,
                face2_points=[[0, 0], [0, 1], [0, 2], [0, 3]],
            )
        )
