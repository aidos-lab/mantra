"""Tests for ``mantra.manifold_types``.

Guards the Klein-bottle renaming: the connected sum ``#^2 RP^2`` is
homeomorphic to the Klein bottle, so the enum member now carries the
``"Klein bottle"`` value instead of ``"#^2 RP^2"``.
"""

from mantra.manifold_types import Manifold2Type, Manifold3Type


def test_klein_bottle_member_replaces_p2_rp2():
    assert Manifold2Type.KLEIN_BOTTLE.value == "Klein bottle"
    # The old member/value must be gone.
    assert not hasattr(Manifold2Type, "P2_RP_2")
    assert "#^2 RP^2" not in {m.value for m in Manifold2Type}


# Every 2-manifold homeomorphism type present in 2_manifolds.json
# (verified against the dataset on 2026-06-11).
DATASET_2M_NAMES = {
    # orientable: connected sums of tori
    "S^2",
    "T^2",
    "#^2 T^2",
    "#^3 T^2",
    "#^4 T^2",
    "#^5 T^2",
    "#^6 T^2",
    "#^8 T^2",
    # non-orientable: connected sums of projective planes
    "RP^2",
    "Klein bottle",
    "#^3 RP^2",
    "#^4 RP^2",
    "#^5 RP^2",
    "#^6 RP^2",
    "#^7 RP^2",
    "#^8 RP^2",
    "#^10 RP^2",
    "#^12 RP^2",
    "#^15 RP^2",
    "#^16 RP^2",
    "#^17 RP^2",
}


def test_enum_covers_exactly_the_dataset_2m_names():
    # The enum must enumerate every type present in the dataset and
    # nothing else (no stale or invented members).
    assert {m.value for m in Manifold2Type} == DATASET_2M_NAMES


def test_orientable_block_precedes_non_orientable_block():
    values = [m.value for m in Manifold2Type]
    # S^2 ... #^8 T^2 come before RP^2 ... #^17 RP^2.
    assert values.index("#^8 T^2") < values.index("RP^2")


# Every 3-manifold homeomorphism type present in 3_manifolds.json
# (verified against the dataset on 2026-06-12).
DATASET_3M_NAMES = {
    "S^3",
    "S^2 x S^1",
    "S^2 twist S^1",
    "RP^3",
    "L(3,1)",
    "L(4,1)",
    "T^3",
    "S^3/Q",
    "(S^2 x S^1)#(S^2 x S^1)",
}


def test_3m_enum_covers_exactly_the_dataset_3m_names():
    assert {m.value for m in Manifold3Type} == DATASET_3M_NAMES
