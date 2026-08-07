"""KLayout PCell for the GP-synthesized octagonal spiral inductor.

Geometry (w, s, d_out, n_turns) is the actual PCell parameter set -- it is
what gets drawn and what DRC sees. Setting `synthesize` re-derives those four
values from a target inductance/frequency/topology via the GP model in
inductor_lab.gp.spiral, the same call the notebook makes; turning it off
leaves w/s/d_out/n_turns free for hand editing.
"""

import math
from typing import TYPE_CHECKING, cast

from .compat import pya
from .geometry import make_45_bridge, oct_fill, oct_segment
from .tech import Layers, TechLayers, active_technology, round_to_grid


def _via_unit_cell(layout: "pya.Layout", via_width: float, via_layer: int) -> "pya.Cell":
    """A single via-cut box, shared by every via array instance in `layout`."""
    name = f"INDUCTOR_LAB_VIA_UNIT_{via_width:.4f}"
    cell = layout.cell(name)
    # The stub declares Layout.cell(str) -> Cell, but it returns None when the
    # name doesn't exist yet (see Layout.cell's own docstring).
    if cell is None:  # pyright: ignore[reportUnnecessaryComparison]
        cell = layout.create_cell(name)
        cell.shapes(via_layer).insert(pya.DBox(0, 0, via_width, via_width))
    return cell


class InductorPCell(pya.PCellDeclarationHelper):
    if TYPE_CHECKING:
        # Materialized dynamically by self.param(...) in __init__ via
        # PCellDeclarationHelper's descriptor-based parameter machinery --
        # declared here only so static analysis knows these attributes exist;
        # the dummy values are never assigned at runtime (TYPE_CHECKING is
        # always False then), they just satisfy "declared but uninitialized".
        synthesize: bool = True
        topology_name: str = ""
        target_l_nh: float = 0.0
        freq_ghz: float = 0.0
        min_srf_ghz: float = 0.0
        w: float = 0.0
        s: float = 0.0
        d_out: float = 0.0
        n_turns: int = 0
        est_l_nh: float = 0.0
        est_q: float = 0.0
        est_fsr_ghz: float = 0.0

    def __init__(self, tl: TechLayers) -> None:
        super().__init__()
        self._tl: TechLayers = tl

        Type = pya.PCellParameterDeclaration
        # Off by default: inductor_lab.gp.spiral imports gpkit at module level,
        # so turning this on is the only thing in this file that requires
        # gpkit to be installed for KLayout's own Python interpreter. The
        # import is deferred to _run_synthesis() so hand-editing w/s/d_out/N
        # works without gpkit at all. Values below must match
        # inductor_lab.gp.spiral.Topology's member values.
        self.param("synthesize", Type.TypeBoolean, "Synthesize w/s/d_out/N from target L (GP)", default=False)
        self.param(
            "topology_name", Type.TypeList, "Topology",
            default="differential",
            choices=[("differential", "differential"), ("single_ended", "single_ended")],
        )
        self.param("target_l_nh", Type.TypeDouble, "Target L [nH]", default=10.0)
        self.param("freq_ghz", Type.TypeDouble, "Operating frequency [GHz]", default=2.4)
        self.param("min_srf_ghz", Type.TypeDouble, "Minimum self-resonance [GHz]", default=7.0)

        self.param("w", Type.TypeDouble, "Track width [um]", default=15.0)
        self.param("s", Type.TypeDouble, "Track spacing [um]", default=2.0)
        self.param("d_out", Type.TypeDouble, "Outer diameter [um]", default=400.0)
        self.param("n_turns", Type.TypeInt, "Number of turns", default=4)

        self.param("est_l_nh", Type.TypeDouble, "GP estimate: L [nH]", default=0.0, readonly=True)
        self.param("est_q", Type.TypeDouble, "GP estimate: Q", default=0.0, readonly=True)
        self.param("est_fsr_ghz", Type.TypeDouble, "GP estimate: f_sr [GHz]", default=0.0, readonly=True)

    # The mixin's default _impl methods each have a single `return <literal>`
    # statement, so pyright infers a Literal return type narrower than the
    # real contract (str / bool); these overrides use the real, wider type.
    def display_text_impl(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return f"Inductor(w={self.w:.3g},s={self.s:.3g},d_out={self.d_out:.3g},N={self.n_turns})"

    def can_create_from_shape_impl(self) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        return False

    def parameters_from_shape_impl(self) -> None:
        pass

    def transformation_from_shape_impl(self) -> "pya.Trans":  # pyright: ignore[reportIncompatibleMethodOverride]
        return pya.Trans()

    def coerce_parameters_impl(self) -> None:
        if self.synthesize:
            self._run_synthesis()

        self.n_turns = max(1, int(round(self.n_turns)))
        self.w = max(self.w, 0.01)
        self.s = max(self.s, 0.01)
        self.d_out = max(self.d_out, 2 * self.w)

    def _run_synthesis(self) -> None:
        from inductor_lab.gp.spiral import Topology, design_inductor  # see __init__: deferred to avoid a hard gpkit dependency

        tech = active_technology(self._tl.name)
        sol = design_inductor(
            self.target_l_nh, self.freq_ghz * 1e9, self._tl.pdk,
            topology=Topology(self.topology_name), min_srf_hz=self.min_srf_ghz * 1e9,
        )
        self.w = round_to_grid(tech, sol.w * 1e6)
        self.s = round_to_grid(tech, sol.s * 1e6)
        self.d_out = round_to_grid(tech, sol.d_out * 1e6)
        self.n_turns = round(sol.n)
        self.est_l_nh = sol.L * 1e9
        self.est_q = sol.Q
        self.est_fsr_ghz = sol.f_sr / 1e9

    def produce_impl(self) -> None:
        # PCellDeclarationHelper's mixin assigns these as plain instance
        # attributes (see pcell_declaration_helper.py); by KLayout's calling
        # contract both are always bound to real objects during produce_impl.
        layout = cast(pya.Layout, self.layout)
        cell = cast(pya.Cell, self.cell)

        tech = active_technology(self._tl.name)
        tl = self._tl
        layers = Layers(layout, tl)

        w, s, r, N = self.w, self.s, self.d_out / 2, self.n_turns
        e = round_to_grid(tech, w + s + (w + 2 * s) / (1 + math.sqrt(2)))
        conn_len = 3 * w

        via = tl.pdk.top_vias[0]
        via_width = via.width * 1e6
        via_spacing = via.space * 1e6
        via_enc = via.enc_upper * 1e6
        via_pitch = via_width + via_spacing

        via_cell = _via_unit_cell(layout, via_width, layers.underpass_via)

        # Segments
        for n in range(N):
            seg = oct_segment(tech, w, r - n * (w + s), e)
            for i in range(4):
                seg.transform(pya.DTrans(rot=45 * i))
                cell.shapes(layers.signal).insert(seg)

        # Bridges, underpasses, and their via arrays
        for n in range(N - 1):
            bridge = make_45_bridge(tech, w, e, s, add_vias=True)
            bridge.transform(pya.DTrans(rot=45))
            bridge.transform(pya.DTrans(0, (-1) ** (2 + n) * (r - (1 + n) * (w + s) + s / 2)))
            upass = bridge.transformed(pya.DTrans(rot=90, mirrx=True))
            cell.shapes(layers.signal).insert(bridge)
            cell.shapes(layers.underpass).insert(upass)

            left_bottom = min(upass.each_point_hull(), key=lambda p: (p.x, p.y))
            right_bottom = max(upass.each_point_hull(), key=lambda p: (p.x, -p.y))
            via_number = math.floor((w - 2 * via_enc + via_spacing) / via_pitch)
            array_size = via_number * via_width + (via_number - 1) * via_spacing
            centering_offset = via_enc + (w - 2 * via_enc - array_size) / 2

            cell.insert(pya.DCellInstArray(
                via_cell.cell_index(),
                pya.DTrans(left_bottom.x + centering_offset, left_bottom.y + centering_offset),
                pya.DVector(via_pitch, 0), pya.DVector(0, via_pitch),
                via_number, via_number,
            ))
            cell.insert(pya.DCellInstArray(
                via_cell.cell_index(),
                pya.DTrans(right_bottom.x - centering_offset - via_width, right_bottom.y + centering_offset),
                pya.DVector(-via_pitch, 0), pya.DVector(0, via_pitch),
                via_number, via_number,
            ))

        # Patch segments together into turns
        for n in range(N):
            patch = pya.DBox(pya.DPoint(r, e / 2), pya.DPoint(r - w, -e / 2))
            patch.move(-n * (w + s), 0)
            cell.shapes(layers.signal).insert(patch)
            cell.shapes(layers.signal).insert(patch.transformed(pya.DTrans(rot=90)))

        # Patch the inner turn
        patch = pya.DBox(pya.DPoint(e / 2, r - (N - 1) * (w + s)), pya.DPoint(-e / 2, r - w - (N - 1) * (w + s)))
        if not (N % 2):
            patch = patch.transformed(pya.DTrans(rot=90))
        cell.shapes(layers.signal).insert(patch)

        # Connections out to the pins
        conn = pya.DBox(pya.DPoint(-e / 2 - w / 2, -r + w), pya.DPoint(-e / 2 + w / 2, -r - conn_len))
        cell.shapes(layers.signal).insert(conn)
        cell.shapes(layers.signal).insert(conn.moved(e, 0))

        # Merge all polygons in the signal layer
        region = pya.Region(cell.begin_shapes_rec(layers.signal))
        region.merge()
        cell.shapes(layers.signal).clear()
        cell.shapes(layers.signal).insert(region)

        # Pins
        pin_height = s
        pin = pya.DBox(
            pya.DPoint(-e / 2 - w / 2, -r - conn_len + pin_height),
            pya.DPoint(-e / 2 + w / 2, -r - conn_len),
        )
        cell.shapes(layers.signal_pin).insert(pin)
        cell.shapes(layers.signal_pin).insert(pin.moved(e, 0))
        if layers.ind_pin is not None:
            cell.shapes(layers.ind_pin).insert(pin)
            cell.shapes(layers.ind_pin).insert(pin.moved(e, 0))

        # No-fill keepout region and inductor marker (only on PDKs that define them)
        keepout = oct_fill(r + conn_len)
        for layer in layers.no_fill:
            cell.shapes(layer).insert(keepout)
        if layers.no_rcx is not None:
            cell.shapes(layers.no_rcx).insert(keepout)
        if layers.ind is not None:
            cell.shapes(layers.ind).insert(keepout)

        # Pin labels (only on PDKs that define a marker text layer)
        if layers.ind_text is not None:
            cell.shapes(layers.ind_text).insert(pya.DText("P1", -e / 2 - w / 2, -r - conn_len))
            cell.shapes(layers.ind_text).insert(pya.DText("P2", e / 2 - w / 2, -r - conn_len))
