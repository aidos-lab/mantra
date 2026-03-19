from enum import Enum


class Manifold2Type(Enum):
    """
    This represents the underlying 2-manifold type,
    which a triangulation is (a triangulation) of.
    """

    S_2 = "S^2"
    T_2 = "T^2"
    RP_2 = "RP^2"
    P2_RP_2 = "#^2 RP^2"
    P3_RP_2 = "#^3 RP^2"
    P4_RP_2 = "#^4 RP^2"
    P5_RP_2 = "#^5 RP^2"


class Manifold3Type(Enum):
    """
    This represents the underlying 3-manifold type,
    which a triangulation is (a triangulation) of.
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
