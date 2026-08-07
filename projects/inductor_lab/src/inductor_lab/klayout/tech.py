"""Per-technology layer maps for the inductor PCell.

KLayout produces PCells registered through a pya.Library in the library's own
internal layout, not the caller's target layout -- self.layout.technology_name
inside produce_impl/coerce_parameters_impl does not reflect the technology the
PCell was actually placed under. So instead, library.py registers one
InductorPCell instance per entry in _TECH_LAYERS, each locked to its own
Library.technology so KLayout shows the right one for whichever PDK is
currently selected. Add an entry here (and it picks up automatically) to
support another PDK.
"""

import math
from dataclasses import dataclass, field

from inductor_lab.pdk.gf180mcu import GF180MCU, GF180MCUParams
from inductor_lab.pdk.sg13g2 import SG13G2, SG13G2Params

from .compat import pya

GdsLayer = tuple[int, int]


@dataclass(frozen=True)
class TechLayers:
    """GDS layer/datatype bindings for one KLayout technology, plus the
    matching inductor_lab.pdk physics parameters."""

    name: str                # KLayout technology name (Library.technology)
    pdk: SG13G2Params | GF180MCUParams
    signal: GdsLayer         # top spiral conductor
    underpass: GdsLayer      # cross-under conductor
    underpass_via: GdsLayer  # via cut between signal and underpass
    pin_datatype: int        # datatype (on the signal layer number) marking pins
    # Optional extras with no known equivalent on every PDK; skipped if None/empty.
    ind: GdsLayer | None = None       # PCell-recognition marker (e.g. SG13G2's RF model binding)
    ind_pin: GdsLayer | None = None
    ind_text: GdsLayer | None = None
    no_fill: tuple[GdsLayer, ...] = field(default_factory=tuple)  # dummy-fill keepout layers
    no_rcx: GdsLayer | None = None    # parasitic-extraction exclusion layer


_TECH_LAYERS: dict[str, TechLayers] = {
    "sg13g2": TechLayers(
        name="sg13g2",
        pdk=SG13G2,
        signal=(134, 0),         # TopMetal2
        underpass=(126, 0),      # TopMetal1
        underpass_via=(133, 0),  # via between TopMetal1/TopMetal2
        pin_datatype=2,
        ind=(27, 0), ind_pin=(27, 2), ind_text=(27, 25),
        no_fill=(
            (1, 23), (5, 23), (8, 23), (10, 23), (30, 23),
            (50, 23), (67, 23), (126, 23), (134, 23),
        ),
        no_rcx=(148, 0),
    ),
    "gf180mcu": TechLayers(
        name="gf180mcu",
        pdk=GF180MCU,
        # 5LM+TM11kA stack (matches inductor_lab.pdk.gf180mcu.GF180MCU): Metal5
        # itself is drawn at TM11kA thickness, there is no separate MetalTop layer
        # above it in this configuration.
        signal=(81, 0),        # Metal5
        underpass=(46, 0),     # Metal4
        underpass_via=(41, 0), # Via4
        # GF180MCU has no dedicated pin layer like SG13G2's TopMetal2.2; this
        # reuses the Metal5_Label datatype as a visual/net marker only.
        pin_datatype=10,
    ),
}


def tech_layers_for(name: str) -> TechLayers:
    try:
        return _TECH_LAYERS[name]
    except KeyError:
        raise ValueError(
            f"No InductorLab layer map for KLayout technology {name!r}; "
            + f"supported: {sorted(_TECH_LAYERS)}"
        ) from None


def supported_technologies() -> list[str]:
    return list(_TECH_LAYERS)


def active_technology(name: str) -> "pya.Technology":
    """The pya.Technology registered under this name.

    Raises if it isn't registered with KLayout (Setup > Technologies, or
    pya.Technology.load + register_technology).
    """
    if name in pya.Technology.technology_names():
        return pya.Technology.technology_by_name(name)
    raise ValueError(
        f"KLayout technology {name!r} is not registered; load it "
        + "(Setup > Technologies, or pya.Technology.load + register_technology) "
        + "before placing the Inductor PCell."
    )


def round_to_grid(tech: "pya.Technology", value: float, floor: bool = False, ceil: bool = False) -> float:
    """Round `value` (in the technology's user units, i.e. um) onto its layout grid."""
    grid = tech.default_grid()
    if grid == 0.0:
        return value
    if floor:
        return math.floor(value / grid) * grid
    if ceil:
        return math.ceil(value / grid) * grid
    return round(value / grid) * grid


def round_point_to_grid(tech: "pya.Technology", x: float, y: float) -> tuple[float, float]:
    return round_to_grid(tech, x), round_to_grid(tech, y)


class Layers:
    """Layer indices for the inductor PCell, bound to a specific pya.Layout."""

    def __init__(self, layout: "pya.Layout", tl: TechLayers):
        self.signal: int = layout.layer(*tl.signal)
        self.underpass: int = layout.layer(*tl.underpass)
        self.underpass_via: int = layout.layer(*tl.underpass_via)
        self.signal_pin: int = layout.layer(tl.signal[0], tl.pin_datatype)

        self.ind: int | None = layout.layer(*tl.ind) if tl.ind else None
        self.ind_pin: int | None = layout.layer(*tl.ind_pin) if tl.ind_pin else None
        self.ind_text: int | None = layout.layer(*tl.ind_text) if tl.ind_text else None
        self.no_fill: list[int] = [layout.layer(*gl) for gl in tl.no_fill]
        self.no_rcx: int | None = layout.layer(*tl.no_rcx) if tl.no_rcx else None
