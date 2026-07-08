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
