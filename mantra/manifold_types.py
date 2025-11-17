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
