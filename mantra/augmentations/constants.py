
# Mapping from 2D manifold names to the name after gluing a torus.
# Based on the classification of closed surfaces:
# - Orientable genus g -> genus g+1
# - Non-orientable with k crosscaps + torus = k+2 crosscaps
TORUS_GLUE_MAP = {
    "S^2": "T^2",
    "T^2": "#^2 T^2",
    "#^2 T^2": "#^3 T^2",
    "#^3 T^2": "#^4 T^2",
    "#^4 T^2": "#^5 T^2",
    "#^5 T^2": "#^6 T^2",
    "#^6 T^2": "#^7 T^2",

    # Non-orientable: torus + k crosscaps = k+2 crosscaps
    "RP^2": "#^3 RP^2",
    "Klein bottle": "#^4 RP^2",
    "#^3 RP^2": "#^5 RP^2",
    "#^4 RP^2": "#^6 RP^2",
    "#^5 RP^2": "#^7 RP^2",
    "#^6 RP^2": "#^8 RP^2",
}

# Mapping after gluing a crosscap (connected sum with RP^2).
# Orientable genus g + crosscap = 2g+1 crosscaps (non-orientable).
# Non-orientable k crosscaps + 1 = k+1 crosscaps.
CROSSCAP_GLUE_MAP = {
    "S^2": "RP^2",
    "T^2": "#^3 RP^2",
    "RP^2": "Klein bottle",
    "Klein bottle": "#^3 RP^2",
    "#^2 T^2": "#^5 RP^2",
    "#^3 RP^2": "#^4 RP^2",
    "#^4 RP^2": "#^5 RP^2",
    "#^3 T^2": "#^7 RP^2",
    "#^5 RP^2": "#^6 RP^2",
    "#^6 RP^2": "#^7 RP^2",
}

# Betti numbers (over Z) for all 2-manifold classes.
# Orientable genus g: [1, 2g, 1]
# Non-orientable k crosscaps: [1, k-1, 0]
BETTI_NUMBERS = {
    "S^2": [1, 0, 1],
    "T^2": [1, 2, 1],
    "#^2 T^2": [1, 4, 1],
    "#^3 T^2": [1, 6, 1],
    "#^4 T^2": [1, 8, 1],
    "#^5 T^2": [1, 10, 1],
    "#^6 T^2": [1, 12, 1],
    "#^7 T^2": [1, 14, 1],
    "#^8 T^2": [1, 16, 1],
    "RP^2": [1, 0, 0],
    "Klein bottle": [1, 1, 0],
    "#^3 RP^2": [1, 2, 0],
    "#^4 RP^2": [1, 3, 0],
    "#^5 RP^2": [1, 4, 0],
    "#^6 RP^2": [1, 5, 0],
    "#^7 RP^2": [1, 6, 0],
    "#^8 RP^2": [1, 7, 0],
}
