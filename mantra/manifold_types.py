from enum import Enum

class Manifold2Type(Enum):
    """
    This represents the underlying 2-manifold type,
    which a triangulation is (a triangulation) of.

    The members enumerate every closed-surface homeomorphism type that
    occurs in the MANTRA 2-manifold dataset.
    """
    # Orientable surfaces: connected sums of tori, ordered by genus.
    S_2 = "S^2"
    T_2 = "T^2"
    P2_T_2 = "#^2 T^2"
    P3_T_2 = "#^3 T^2"
    P4_T_2 = "#^4 T^2"
    P5_T_2 = "#^5 T^2"
    P6_T_2 = "#^6 T^2"
    P7_T_2 = "#^7 T^2"
    P8_T_2 = "#^8 T^2"

    # Non-orientable surfaces: connected sums of projective planes,
    # ordered by genus (#^2 RP^2 is the Klein bottle).
    RP_2 = "RP^2"
    KLEIN_BOTTLE = "Klein bottle"
    P3_RP_2 = "#^3 RP^2"
    P4_RP_2 = "#^4 RP^2"
    P5_RP_2 = "#^5 RP^2"
    P6_RP_2 = "#^6 RP^2"
    P7_RP_2 = "#^7 RP^2"
    P8_RP_2 = "#^8 RP^2"
    P10_RP_2 = "#^10 RP^2"
    P12_RP_2 = "#^12 RP^2"
    P15_RP_2 = "#^15 RP^2"
    P16_RP_2 = "#^16 RP^2"
    P17_RP_2 = "#^17 RP^2"

class Manifold3Type(Enum):
    """
    This represents the underlying 3-manifold type,
    which a triangulation is (a triangulation) of.

    The members enumerate every 3-manifold homeomorphism type that
    occurs in the MANTRA 3-manifold dataset.
    """

    S_3 = "S^3"
    S2_x_S1 = "S^2 x S^1"
    S2_twist_S1 = "S^2 twist S^1"
    RP_3 = "RP^3"
    L_3_1 = "L(3,1)"
    L_4_1 = "L(4,1)"
    T_3 = "T^3"
    S3_Q = "S^3/Q"
    S2xS1_hash_S2xS1 = "(S^2 x S^1)#(S^2 x S^1)"
