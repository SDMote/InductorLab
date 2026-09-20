"""
Shared dataclasses for process (PDK) parameters.

Every process module (sg13g2.py, ...) builds one ProcessParams instance from
MetalLayer/ViaLayer and exposes it as a module-level constant. GP/physics
code takes a ProcessParams as a plain function argument and never imports a
specific process by name -- only entry points (CLI, layout export) do that.
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
    thickness: float  # via height [m]
    width: float      # via size [m]
    space: float      # min space between vias [m]
    enc_lower: float  # min enclosure in the metal below [m]
    enc_upper: float  # min enclosure in the metal above [m]


@dataclass(frozen=True)
class ProcessParams:
    """Technology constants for one process, shared shape across all PDKs.

    top_metals and top_vias are ordered top-to-bottom, so index 0 is always
    the topmost layer (the preferred inductor conductor).
    """
    name: str  # short process id, e.g. "sg13g2" -- matches the module name

    top_metals: tuple[MetalLayer, ...]
    top_vias: tuple[ViaLayer, ...]

    # -- Oxide dielectric (BEOL stack) --------------------------------------
    eps_r_ox: float  # relative permittivity (dimensionless)
    t_ox: float      # oxide thickness, substrate surface to top metal [m]

    # -- Substrate ------------------------------------------------------------
    eps_r_sub: float  # relative permittivity (dimensionless)
    sigma_sub: float  # conductivity [S/m]
    t_sub: float      # thickness [m]
