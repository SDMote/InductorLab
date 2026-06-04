"""
Shared dataclasses and sentinels for PDK layer parameters.

Import MetalLayer, ViaLayer, and TODO from here in every process file so that
the GP modules have a single stable type to program against.
"""

import math
from dataclasses import dataclass

TODO: float = math.nan  # sentinel for parameters not found in available documentation


@dataclass(frozen=True)
class MetalLayer:
    """Electrical and DRC parameters for a single metal layer."""
    name: str
    sigma: float      # conductivity [S/m]
    thickness: float  # [m]
    w_min: float      # DRC min width [m]
    s_min: float      # DRC min space or notch [m]


@dataclass(frozen=True)
class ViaLayer:
    """Electrical and DRC parameters for a single via layer."""
    name: str
    sigma: float      # conductivity [S/m]
    thickness: float  # oxide layer thickness (= via height) [m]
    width: float      # via size [m]  (fixed: TV*.a is both min and max)
    space: float      # min space between vias [m]
    enc_lower: float  # min enclosure in the metal below [m]
    enc_upper: float  # min enclosure in the metal above [m]
