#!/usr/bin/env python3
"""
simulate_rapidfem.py
====================
Simulate ONE symmetric octagonal inductor with the rapidfem FEM solver
(https://github.com/milanofthe/rapidfem) as a fast alternative to Palace.

It reuses the *exact* GDS that the Palace flow already produced
(``<model_dir>/<ind_name>_forEM.gds`` from scripts/generate_sweep_palace.py),
so the two solvers see identical geometry and the comparison is apples-to-apples.
The result is written as a Touchstone ``.s2p`` into ``<model_dir>/rapidfem/`` --
the same per-model-dir layout the downstream extraction (scripts/fit_cbr.best_s2p,
analyze_lhs.py) already globs for, so no post-processing changes are needed.

Layer/port model (matches generate_sweep_palace.build_one + Palace setup):
  - signal coil on TopMetal2 (134/0), underpass on TopMetal1 (126/0),
    underpass vias on TopVia2 (133/0): all -> PEC, extruded by from_gds via the
    built-in rfic.Stack.sg13g2() (gds numbers line up 1:1).
  - ground frame on Metal1 (8/0) -> PEC reference.
  - two lumped ports: vertical plates bridging Metal1 (z_top) up to TopMetal2
    (z_bottom) at the 201/202 Palace port markers, direction +z, Z0 = 50 ohm.

Usage:
    python3 scripts/simulate_rapidfem.py <model_dir_or_gds> [--freqs 2.5e9 ...]
    python3 scripts/simulate_rapidfem.py \
        scripts/.out/sweep_2p5ghz/sweep_L4_N3_33p805_2p0_423p2 --freqs 2.5e9
If a directory is given, the unique *_forEM.gds inside it is used.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import klayout.db as db
import numpy as np

import rapidfem as rf
import rapidfem.rfic as rfic

# Palace port-marker GDS layers written by generate_sweep_palace.build_one().
PORT_LAYERS = [(201, 0), (202, 0)]
GND_LAYER = (8, 0)          # Metal1 ground frame
TRACE_LAYER_NAMES = ("TopMetal2", "TopMetal1", "TopVia2")
GND_LAYER_NAME = "Metal1"
DEFAULT_FREQS = [2.5e9]
UM = 1e-6
REPO = Path(__file__).resolve().parents[2]   # scripts/rapidfem/<file> -> repo root


def find_gds(target: Path) -> Path:
    if target.is_file() and target.suffix == ".gds":
        return target
    if target.is_dir():
        cands = sorted(target.glob("*_forEM.gds"))
        if not cands:
            sys.exit(f"no *_forEM.gds in {target}")
        if len(cands) > 1:
            sys.exit(f"ambiguous: {len(cands)} *_forEM.gds in {target}: {cands}")
        return cands[0]
    sys.exit(f"not a .gds file or directory: {target}")


def load_stack_from_xml(xml_path: Path):
    """Build a rapidfem rfic.Stack from the gds2palace SG13G2 XML stackup so
    rapidfem and Palace use the IDENTICAL stack (z-positions, thickness, sigma,
    er). The built-in rfic.Stack.sg13g2() is only an approximation -- its
    Metal1/TopMetal2 z and substrate sigma (20 vs the XML's 2.0 S/m) differ.
    """
    import xml.etree.ElementTree as ET
    um = 1e-6
    root = ET.parse(xml_path).getroot()
    mats = {m.get("Name"): m for m in root.iter("Material")}
    layers = []
    for L in root.iter("Layer"):
        typ = L.get("Type", "").lower()
        if typ not in ("conductor", "via"):
            continue
        gds = int(L.get("Layer"))
        zmin, zmax = float(L.get("Zmin")), float(L.get("Zmax"))
        mat = mats.get(L.get("Material"))
        sigma = float(mat.get("Conductivity")) if mat is not None else 0.0
        layers.append(rfic.PdkLayer(
            name=L.get("Name"), gds=gds, datatype=0,
            z=zmin * um, thickness=(zmax - zmin) * um,
            type=("via" if typ == "via" else "metal"), sigma=sigma))
    sub = mats.get("Substrate")
    sio2 = mats.get("SiO2")
    return rfic.Stack(
        name="SG13G2_XML", layers=layers,
        substrate_thickness=200 * um,
        substrate_er=float(sub.get("Permittivity")) if sub is not None else 11.9,
        substrate_sigma=float(sub.get("Conductivity")) if sub is not None else 2.0,
        oxide_er=float(sio2.get("Permittivity")) if sio2 is not None else 4.1,
        oxide_tand=0.0)


def preprocess_gds(gds: Path, stack, cellname="Inductor", drop_layers=()) -> Path:
    """Rewrite the coil GDS so rapidfem.from_gds can extrude it.

    InductorLab's GDS stores each metal as ONE merged klayout Region, often a
    polygon WITH HOLES (the N=1 coil is an annulus; the Metal1 ground is a
    frame). rapidfem's from_gds extrudes each gdstk polygon via a single OCC
    curve loop and chokes on holes / keyhole bridges ("curve loop not closed").
    We merge each stack layer's shapes and decompose them into simple, hole-free
    trapezoids -- from_gds extrudes those cleanly and `merge=True` fragments
    them back into one conformal conductor. Returns a temp .gds path.
    """
    ly = db.Layout(); ly.read(str(gds))
    cell = ly.cell(cellname)
    if cell is None:
        sys.exit(f"cell {cellname!r} not in {gds}")
    dbu = ly.dbu

    out = db.Layout(); out.dbu = dbu
    ocell = out.create_cell(cellname)
    n_polys = 0
    for pl in stack.layers:
        if pl.name in drop_layers:
            continue
        li = ly.find_layer(pl.gds, pl.datatype)
        if li is None:
            continue
        reg = db.Region(cell.begin_shapes_rec(li)); reg.merge()
        if reg.is_empty():
            continue
        # Via arrays are hundreds of tiny squares; a morphological close
        # (size up by the via pitch, then back) coalesces each array into a few
        # solid blocks -- electrically equivalent for EM, but a handful of mesh
        # volumes instead of ~1000 sliver boxes that would OOM the mesher.
        if pl.type == "via":
            grow = int(round(2.0 / dbu))   # ~2 um, > via spacing
            reg = reg.sized(grow).merged().sized(-grow)
        oli = out.layer(pl.gds, pl.datatype)
        for trap in reg.decompose_trapezoids().each():
            ocell.shapes(oli).insert(trap)
            n_polys += 1
    tmp = Path(tempfile.mkdtemp(prefix="rapidfem_")) / (gds.stem + "_simple.gds")
    out.write(str(tmp))
    print(f"prep:   {n_polys} simple polygons -> {tmp.name}")
    return tmp


def read_port_markers(gds: Path, cellname="Inductor"):
    """Return list of (cx, cy, w_x) in metres for each Palace port marker box.

    The marker is a thin box on TopMetal2 at the coil's pin; its x-extent is the
    conductor width, its centre gives the port plate location.
    """
    ly = db.Layout()
    ly.read(str(gds))
    cell = ly.cell(cellname)
    if cell is None:
        sys.exit(f"cell {cellname!r} not in {gds}")
    ports = []
    for (lnum, dt) in PORT_LAYERS:
        li = ly.find_layer(lnum, dt)
        if li is None:
            sys.exit(f"port marker layer {lnum}/{dt} missing in {gds}")
        bxs = [sh.dbbox() for sh in cell.shapes(li).each()]
        if len(bxs) != 1:
            sys.exit(f"expected 1 box on {lnum}/{dt}, found {len(bxs)}")
        b = bxs[0]
        ports.append((b.center().x * UM, b.center().y * UM, b.width() * UM))
    return ports


def load_dielectric_stack_from_xml(xml_path):
    """Parse the full <Dielectrics> stack so rapidfem sees Palace's ACTUAL dielectric
    environment (Substrate/EPI/SiO2/Passive/AIR), not just the simplified
    silicon+SiO2 of create_substrate.

    gds2palace stacks the dielectrics from z=0 upward in reversed document order
    (calculate_zpositions) and instead shifts the METAL layers up by
    <Substrate Offset>; in rapidfem's native coords (z=0 at the silicon surface,
    where load_stack_from_xml reads the metals) that is equivalent to subtracting
    the offset from the dielectric z's. Returns a list of dicts
    {name, er, sigma, zmin, zmax} (metres), bottom-to-top.

    Notably this puts conductive silicon (Substrate sigma=2, EPI sigma=5) with its
    surface at z=0 -- ~11 um below TopMetal2 -- which create_substrate did NOT
    (it filled everything below the coil with lossless SiO2 and pushed the silicon
    far below). The near, lossy silicon is a partial image/eddy plane that can
    lower L toward Palace.
    """
    import xml.etree.ElementTree as ET
    um = 1e-6
    root = ET.parse(xml_path).getroot()
    mats = {m.get("Name"): m for m in root.iter("Material")}
    offset = 0.0
    for s in root.iter("Substrate"):
        if s.get("Offset") is not None:
            offset = float(s.get("Offset"))
    diels = list(root.iter("Dielectric"))   # document order = top..bottom
    # stack from z=0 upward in REVERSED order (matches gds2palace)
    z = 0.0
    placed = []
    for d in reversed(diels):
        t = float(d.get("Thickness"))
        placed.append((d, z, z + t))
        z += t
    out = []
    for d, zmin, zmax in placed:
        m = mats.get(d.get("Material"))
        er = float(m.get("Permittivity")) if m is not None else 1.0
        sigma = float(m.get("Conductivity")) if m is not None else 0.0
        out.append(dict(name=d.get("Name"), er=er, sigma=sigma,
                        zmin=(zmin - offset) * um, zmax=(zmax - offset) * um))
    return out


def make_stack(stack_xml):
    if stack_xml == "builtin":
        return rfic.Stack.sg13g2()
    p = Path(stack_xml)
    return load_stack_from_xml(p if p.is_absolute() else REPO / p)


def weld_vias_to_thin_plates(stack):
    """Lower each via's BOTTOM onto the 2-D plate it must connect to.

    In thin_conductors mode `from_gds` lays every metal as a zero-thickness plate
    at the layer's BOTTOM z (`pdk.z`). A via's physical span is
    [lower_metal.z_top, upper_metal.z]: its TOP lands exactly on the upper plate
    (`upper_metal.z`, good) but its BOTTOM sits at `lower_metal.z_top` -- one
    metal-thickness ABOVE the lower plate at `lower_metal.z`. The via then never
    welds to the lower metal, so every underpass crossunder is electrically OPEN
    and multi-turn coils read a capacitive / NEGATIVE inductance (the port sees a
    gap, not a coil). Fix: extend each via down to `lower_metal.z` (keeping its
    top), so its bottom face is coincident with the lower plate and fragment welds
    them. Mutates the stack in place; thin-conductor mode only (in conductors_3d
    mode the metals are real volumes and the via already meets their faces).
    Idempotent: once a via bottom sits at a metal's z (not its z_top) no metal
    matches and it is left alone.
    """
    metals = list(stack.metals())
    fixed = []
    for v in stack.vias():
        lower = [m for m in metals if abs(m.z_top - v.z) < 1e-9]
        if not lower:
            continue                      # via bottom already on a plate (or no match)
        m = lower[0]
        z_top = v.z + v.thickness         # unchanged: still meets the upper plate
        v.z = m.z
        v.thickness = z_top - m.z
        fixed.append((v.name, m.name, v.thickness * 1e6))
    if fixed:
        print("via-weld: " + ", ".join(
            f"{vn} -> {mn} bottom (t={t:.2f}um)" for vn, mn, t in fixed))
    return stack


def build_model(gds: Path, stack, *, pad=1.1, air_thick=30.0, maxh=120.0,
                metal_maxh=None, drop_layers=(), use_pec=False, ground="frame",
                port_mode="ground", conductors_3d=False, dielectric_stack=None,
                backside_ground=False):
    """Build the full rapidfem Geometry (coil + substrate + air + lumped ports +
    BCs + local refinement) from a Palace *_forEM.gds, ready for g.mesh(). Shared
    by the solver (main) and the viewer (scripts/rapidfem_view.py). Returns
    (g, port_plates, info).

    ground:
      "frame" -- keep the GDS Metal1 ground frame (matches Palace's GDS).
      "strip" -- drop the big frame, connect the two port ground pads with a small
                 Metal1 strip. The frame is only a port *reference*; as a 2-D sheet
                 it magnetically shields the coil and kills ~9x of the inductance,
                 so this tests the coil with a minimal, non-shielding return.
      "none"  -- drop the frame, no extra return (ports float -- debug only).
    """
    ports = read_port_markers(gds)
    if conductors_3d:
        # Real 3-D conductors: the vertical port plate must span the OXIDE GAP
        # between the Metal1 top face and the TopMetal2 bottom face (Palace's port).
        z_gnd = stack.by_name(GND_LAYER_NAME).z_top
        z_trace = stack.by_name("TopMetal2").z
    else:
        z_trace = stack.by_name("TopMetal2").z
        # Thin-plate metals sit at each layer's bottom z; lower the vias onto those
        # plates so the underpass crossunders actually weld (else multi-turn coils
        # read capacitive/negative L -- see weld_vias_to_thin_plates).
        weld_vias_to_thin_plates(stack)
        # GROUND-PLANE thin sheet -> place it at the metal's TOP face, not its
        # bottom. from_gds defaults every thin metal to its bottom z, but the
        # field-facing surface of a ground plane (and Palace's port reference) is
        # its top. Anchoring here makes the port span Metal1-top -> TopMetal2-bottom
        # = 9.77 um, exactly Palace's port (vs the 10.19 um bottom-anchored gap),
        # and sets the coil->ground spacing to the true metal-face separation.
        # thickness is preserved (z_top moves up, but nothing in thin mode reads
        # Metal1's z_top for these coils).
        gnd_pl = stack.by_name(GND_LAYER_NAME)
        gnd_pl.z = gnd_pl.z_top              # move the sheet up by one Metal1 thickness
        z_gnd = gnd_pl.z

    if ground in ("strip", "none") or port_mode in ("diff", "backside"):
        drop_layers = tuple(set(drop_layers) | {GND_LAYER_NAME})
    if port_mode == "backside":
        backside_ground = True               # ports reference the backside ground plane
    simple_gds = preprocess_gds(gds, stack, drop_layers=drop_layers)
    # thin_conductors=True: metals become 2D PEC plates (t << w here), avoiding
    # the sliver tets that sub-micron-thick extruded slabs over a ~mm footprint
    # would otherwise force -- this is what makes the solve tractable.
    g = rf.Geometry.from_gds(str(simple_gds), stack=stack, top_cell="Inductor",
                             thin_conductors=not conductors_3d)
    metals = [o for o in g._objects if o.dim == 2]   # 2-D conductor plates (thin mode)
    # Assign finite-conductivity Conductor material to every 3-D conductor volume.
    # In thin_conductors mode only the via layers are 3-D (they MUST get a material
    # or the assembler panics with a singular er=0 tensor). In conductors_3d mode
    # ALL metals are 3-D volumes -- TEST A: real thick conductors like Palace, so
    # the sub-skin-depth thin ground is magnetically transparent (B penetrates) and
    # the coil flux isn't shielded, instead of the zero-thickness-sheet model.
    mm = (metal_maxh if metal_maxh is not None else maxh / 4) * UM
    metal_names_all = {l.name for l in stack.metals()}
    via_names = {l.name for l in stack.vias()}
    for o in g._objects:
        nm = o._entity.name
        if o.dim == 3 and (nm in via_names or nm in metal_names_all):
            # maxh refines the conductor mesh (incl. through the thin slab) so the
            # penetrating field is resolved; the Conductor material carries the loss.
            o.material = rf.Conductor(conductivity=stack.by_name(nm).sigma, maxh=mm)

    # create_substrate / our air box are both centred on the origin, so build a
    # footprint symmetric about (0,0) that still covers the coil's full extent
    # (the symmetric layouts sit ~centred but not exactly, hence max(|min|,|max|)).
    ly = db.Layout(); ly.read(str(gds))
    ibb = ly.cell("Inductor").dbbox()
    wx = 2 * max(abs(ibb.left), abs(ibb.right)) * UM * pad
    wy = 2 * max(abs(ibb.bottom), abs(ibb.top)) * UM * pad

    bsg = None                               # optional explicit backside ground plate
    if dielectric_stack is not None:
        # Full Palace dielectric environment: one box per <Dielectric> layer
        # (Substrate, EPI, SiO2, Passive, AIR) at its true z with its true er/sigma.
        # The top AIR layer is capped at air_thick (the XML's 200 um would bloat the
        # mesh) and is the box whose outer faces carry the ABC.
        diel_boxes, air = [], None
        for d in dielectric_stack:
            zmin, zmax = d["zmin"], d["zmax"]
            h = (air_thick * UM) if d["name"] == "AIR" else (zmax - zmin)
            if d["sigma"] == 0 and abs(d["er"] - 1.0) < 1e-9:
                mat = rf.Air()
            else:
                mat = rf.Dielectric(er=d["er"], conductivity=d["sigma"])
            box = g.box(wx, wy, h, position=(-wx / 2, -wy / 2, zmin), material=mat)
            box.name = d["name"]
            diel_boxes.append(box)
            if d["name"] == "AIR":
                air = box
        if air is None:                      # stack had no explicit AIR cap
            ztop = max(d["zmax"] for d in dielectric_stack)
            air = g.box(wx, wy, air_thick * UM,
                        position=(-wx / 2, -wy / 2, ztop), material=rf.Air())
            diel_boxes.append(air)
        subs = None
        if backside_ground:
            # Explicit PEC ground plane at the substrate backside (deepest dielectric
            # bottom) -- the real SG13G2 BACKSIDEGND. Gives C_si a defined return so
            # the shunt cap / SRF is physical, instead of relying on the default-PEC
            # substrate-bottom boundary. Use with --port diff (no frame).
            zb = min(d["zmin"] for d in dielectric_stack)
            bsg = g.xy_plate(wx, wy, position=(-wx / 2, -wy / 2, zb))
            bsg.name = "BACKSIDEGND"
    else:
        # Simplified silicon + SiO2 (create_substrate). Don't auto-fragment (the 2D
        # metal plates aren't 3D, so we batch them into one fragment call below).
        subs = stack.create_substrate(g, footprint=(wx, wy), center=True,
                                      fragment_existing=False)
        air = g.box(wx, wy, air_thick * UM,
                    position=(-wx / 2, -wy / 2, stack.top_z), material=rf.Air())
        diel_boxes = [subs["oxide"], subs["substrate"], air]

    # Lumped ports. A bare vertical plate whose top/bottom edges merely lie in
    # the INTERIOR of the big coil / ground sheets doesn't reliably weld during
    # fragment, leaving the port floating (open circuit). Following rapidfem's
    # trace_port / spiral example, give each port its own small extension pad on
    # TopMetal2 (welds into the coil) and ground pad on Metal1 (welds into the
    # frame), each sized to the port -- so the port plate's top/bottom edges are
    # genuine pad boundary edges that share DOFs with PEC on both ends.
    port_plates, extra_pads = [], []
    if port_mode == "diff":
        # ONE differential lumped port bridging the two TopMetal2 leads across the
        # gap between them (field direction x), NO ground reference. Measures the
        # coil's differential impedance directly as a 1-port: L = Im(Zin)/w. This
        # isolates the coil from the common-mode-dominated 2-ports-to-ground setup.
        (x1, _, w1), (x2, y2, w2) = ports[0], ports[1]
        gl, gr = x1 + w1 / 2, x2 - w2 / 2          # inner edges of the two leads
        if gl > gr:
            gl, gr = gr, gl
        ph = max(1.5 * (gr - gl), 80 * UM)          # orthogonal extent > gap (width>=height)
        plate = g.xy_plate(gr - gl, ph, position=(gl, y2 - ph / 2, z_trace))
        plate.name = "pdiff"
        port_plates.append(plate)
    elif port_mode == "backside":
        # 2-port referenced to the backside ground, NO frame. The SIGNAL port is
        # SHORT -- a valid lumped gap from the coil terminal (TopMetal2) down to a
        # local ground pad at z_gnd (~Metal1 level). The long path to the backside
        # is a GROUND PLUG (a grounded through-substrate column tied to the backside
        # plane): being ground, its substrate coupling is the desired C_si, not a
        # signal parasitic. (A 195um lumped SIGNAL port is invalid -- it's a
        # distributed feed, not a delta-gap, and reads -1/f^2 garbage; keeping only
        # the ground long fixes that.) Niknejad 2-port extraction applies.
        zbk = min(d["zmin"] for d in dielectric_stack)
        for i, (pcx, pcy, pw) in enumerate(ports, 1):
            ext = g.xy_plate(pw, pw, position=(pcx - pw / 2, pcy - pw / 2, z_trace))
            ext.name = "TopMetal2"
            gtop = g.xy_plate(pw, pw, position=(pcx - pw / 2, pcy - pw / 2, z_gnd))
            gtop.name = "BACKSIDEGND"                       # local ground pad (port ref)
            plug = g.plate(p0=(pcx - pw / 2, pcy, zbk), width=(pw, 0, 0),
                           height=(0, 0, z_gnd - zbk))       # ground plug -> backside
            plug.name = "BACKSIDEGND"
            extra_pads += [ext, gtop, plug]
            plate = g.plate(p0=(pcx - pw / 2, pcy, z_gnd), width=(pw, 0, 0),
                            height=(0, 0, z_trace - z_gnd))  # SHORT signal port
            plate.name = f"p{i}"
            port_plates.append(plate)
    else:
        for i, (pcx, pcy, pw) in enumerate(ports, 1):
            # In 3-D mode the port plate's top/bottom edges land directly on the
            # TopMetal2 and Metal1 conductor faces, so no 2-D welding pads are needed.
            if not conductors_3d:
                ext = g.xy_plate(pw, pw, position=(pcx - pw / 2, pcy - pw / 2, z_trace))
                ext.name = "TopMetal2"
                gnd = g.xy_plate(pw, pw, position=(pcx - pw / 2, pcy - pw / 2, z_gnd))
                gnd.name = "Metal1"
                extra_pads += [ext, gnd]
            plate = g.plate(p0=(pcx - pw / 2, pcy, z_gnd), width=(pw, 0, 0),
                            height=(0, 0, z_trace - z_gnd))
            plate.name = f"p{i}"
            port_plates.append(plate)

    # Minimal-ground return: a small Metal1 strip welding the two port ground pads
    # together (replaces the big shielding frame). Runs along the pads' shared edge,
    # well below the octagon loop, so it references the ports without imaging the coil.
    if ground == "strip" and len(ports) >= 2:
        xs = [pcx for pcx, _, _ in ports]
        pw0 = ports[0][2]
        ystrip = sum(pcy for _, pcy, _ in ports) / len(ports)
        x0, x1 = min(xs) - pw0 / 2, max(xs) + pw0 / 2
        strip = g.xy_plate(x1 - x0, pw0, position=(x0, ystrip - pw0 / 2, z_gnd))
        strip.name = GND_LAYER_NAME
        extra_pads.append(strip)

    # ONE conformal fragment: air + oxide/substrate + every conductor (2-D plates
    # in thin mode, OR 3-D metal/via volumes in 3-D mode) + port pads + port plates,
    # so all interfaces share a conformal mesh.
    #   CRITICAL: the AIR box must be in this fragment. If it is left out, the
    #   oxide-top and air-bottom faces are coincident but UNWELDED, so the oxide's
    #   top face is an *exterior* face and defaults to PEC -- a perfect magnetic
    #   image plane sitting ~3 um above the TopMetal2 coil that shorts its flux and
    #   collapses the extracted L ~6-10x (the long-standing SG13G2 under-read,
    #   localised with a controlled clean-loop morph from a free-air single-turn loop).
    metal_vols = [o for o in g._objects
                  if o.dim == 3 and o._entity.name in (metal_names_all | via_names)]
    g.fragment(*diel_boxes, *metals, *metal_vols, *extra_pads, *port_plates,
               *( [bsg] if bsg is not None else [] ))

    from rapidfem.geometry import EntityCollection
    if conductors_3d:
        # 3-D conductors carry their loss + field penetration via the Conductor
        # material assigned above; no surface BC needed. Nothing else to wire.
        pass
    else:
        # Conductor BC must cover EVERY conductor face. fragment splits the metal
        # plates (the coil face gets imprinted by the port plate) and a GeoObject
        # only tracks ONE representative face afterwards -- so passing the
        # GeoObjects would miss the split-off slivers at the port contact and leave
        # the port open. Instead select all CURRENT dim-2 entities by metal layer
        # name (fragment propagates the name to every child face) = the full set.
        by_layer = {}
        for e in g._entities:
            if e.dim == 2 and any(e.name == l.name for l in stack.metals()):
                by_layer.setdefault(e.name, []).append(e)
        for name, ents in by_layer.items():
            coll = EntityCollection(g, ents)
            if use_pec:
                # Lossless PEC. WARNING: a 2-D PEC sheet is a PERFECT magnetic
                # shield, so a thin ground (t<skin depth) wrongly images the coil.
                rf.PEC(coll)
            else:
                # Finite-conductivity thin-sheet surface impedance: real series R.
                # NOTE: on a zero-thickness embedded sheet this still reads ~10x low
                # L (see memory) -- that is why conductors_3d (Test A) exists.
                pl = stack.by_name(name)
                rf.SurfaceImpedance(coll, conductivity=pl.sigma, thickness=pl.thickness)
            coll.maxh = mm

    if bsg is not None:
        # PEC the explicit backside ground plane (its current dim-2 faces post-fragment).
        from rapidfem.geometry import EntityCollection
        rf.PEC(EntityCollection(g, [e for e in g._entities
                                    if e.dim == 2 and e.name == "BACKSIDEGND"]))

    port_dir = (1, 0, 0) if port_mode == "diff" else (0, 0, 1)
    for plate in port_plates:
        rf.LumpedPort(plate, direction=port_dir, z0=50.0)
        plate.faces.maxh = mm
    rf.ABC(*air.faces.outer)
    return g, port_plates, {"ports": ports, "footprint": (wx, wy)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="model dir containing *_forEM.gds, or a .gds path")
    ap.add_argument("--freqs", type=float, nargs="+", default=DEFAULT_FREQS,
                    help="frequencies [Hz] (default 2.5e9)")
    ap.add_argument("--maxh", type=float, default=40.0, help="global mesh size cap [um]")
    ap.add_argument("--metal-maxh", type=float, default=None,
                    help="finer local mesh size [um] on conductor + port faces -- "
                         "refine where the fields concentrate without blowing up the "
                         "bulk substrate/air mesh (default: maxh/4)")
    ap.add_argument("--pad", type=float, default=1.2,
                    help="footprint = coil bbox * pad (lateral air margin)")
    ap.add_argument("--air-thick", type=float, default=50.0,
                    help="air box height above the stack [um]")
    ap.add_argument("--mesh-only", action="store_true",
                    help="stop after meshing and report tet count (no solve)")
    ap.add_argument("--gmsh-gui", action="store_true",
                    help="open the gmsh GUI to inspect the mesh after meshing (no solve)")
    ap.add_argument("--stack-xml", default="resources/SG13G2_200um.xml",
                    help="gds2palace XML stackup to match Palace exactly; "
                         "pass 'builtin' to use rfic.Stack.sg13g2()")
    ap.add_argument("--drop-layers", default="",
                    help="comma-separated layer NAMES to omit (e.g. 'Metal1' to "
                         "test whether the big PEC ground plane is shielding the coil)")
    ap.add_argument("--pec", action="store_true",
                    help="model conductors as lossless PEC instead of finite-"
                         "conductivity SurfaceImpedance (A/B debugging only)")
    ap.add_argument("--conductors-3d", action="store_true",
                    help="TEST A: extrude metals as real 3-D Conductor(sigma) volumes "
                         "(field penetrates the thin ground like Palace) instead of "
                         "zero-thickness sheets. Faithful but heavier mesh.")
    ap.add_argument("--ground", choices=("frame", "strip", "none"), default="frame",
                    help="ground return: 'frame' = GDS Metal1 frame (Palace setup); "
                         "'strip' = minimal non-shielding strip between port pads; "
                         "'none' = floating (debug)")
    ap.add_argument("--port", choices=("ground", "diff", "backside"), default="ground",
                    help="'ground' = two Metal1->TopMetal2 ports (2-port, Palace-style); "
                         "'diff' = one differential port bridging the two leads (1-port); "
                         "'backside' = two ports TopMetal2->backside ground plane (2-port, "
                         "NO frame; auto-enables --backside-ground). Niknejad applies.")
    ap.add_argument("--backside-ground", action="store_true",
                    help="add an explicit PEC ground plane at the substrate backside "
                         "(the real SG13G2 BACKSIDEGND) so C_si/SRF is physical. Use "
                         "with --port diff (no frame).")
    ap.add_argument("--simple-substrate", action="store_true",
                    help="use the 2-box create_substrate (silicon+SiO2) instead of "
                         "the full Palace <Dielectrics> stack (Substrate/EPI/SiO2/"
                         "Passive/AIR with true z & sigma). Default: full Palace stack.")
    ap.add_argument("--out", default=None, help="output .s2p path")
    a = ap.parse_args()

    gds = find_gds(Path(a.target))
    ind_name = gds.name.replace("_forEM.gds", "")
    model_dir = gds.parent
    out = Path(a.out) if a.out else model_dir / "rapidfem" / f"{ind_name}_rapidfem.s2p"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"GDS:    {gds}")
    print(f"freqs:  {[f/1e9 for f in a.freqs]} GHz")
    stack = make_stack(a.stack_xml)
    print(f"stack:  {a.stack_xml}")

    drop_layers = tuple(s for s in a.drop_layers.split(",") if s)
    diel = None
    if not a.simple_substrate and a.stack_xml != "builtin":
        xp = Path(a.stack_xml)
        diel = load_dielectric_stack_from_xml(xp if xp.is_absolute() else REPO / xp)
        print("diel:   " + ", ".join(
            f"{d['name']}(er={d['er']},s={d['sigma']},z[{d['zmin']*1e6:.1f},{d['zmax']*1e6:.1f}])"
            for d in diel))
    print(f"metals: {'PEC' if a.pec else 'SurfaceImpedance (finite sigma + thickness)'}"
          f"  ground={a.ground}")
    g, port_plates, info = build_model(
        gds, stack, pad=a.pad, air_thick=a.air_thick, maxh=a.maxh,
        metal_maxh=a.metal_maxh, drop_layers=drop_layers, use_pec=a.pec,
        ground=a.ground, port_mode=a.port, conductors_3d=a.conductors_3d,
        dielectric_stack=diel, backside_ground=a.backside_ground)
    print(f"ports:  {[(round(x*1e6,1), round(y*1e6,1)) for x, y, _ in info['ports']]} um")

    g.mesh(maxh=a.maxh * UM)
    # Cheap tet count straight from the gmsh session (rapidfem shares the
    # singleton), so we can gauge problem size before the memory-heavy solve.
    import gmsh
    n_tets = len(gmsh.model.mesh.getElementsByType(4)[0])
    print(f"mesh:   {n_tets} tets (maxh={a.maxh} um, pad={a.pad}, air={a.air_thick} um)")
    print("  physical groups (dim, tag, name, #entities):")
    for dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        ents = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        print(f"    dim={dim} tag={tag} name={name!r} nent={len(ents)}")
    if a.gmsh_gui:
        # Open the gmsh GUI to inspect the mesh/geometry (blocks until closed). No solve.
        print("opening gmsh GUI (close the window to continue)...")
        gmsh.fltk.run()
        return
    if a.mesh_only:
        # Edge-connectivity of each conductor group: a coil broken into pieces
        # (trapezoids not welded by fragment) reads as multiple components and
        # would explain a too-low inductance.
        def n_components(group_name):
            ggroups = {gmsh.model.getPhysicalName(d, t): (d, t)
                       for d, t in gmsh.model.getPhysicalGroups()}
            if group_name not in ggroups:
                return None
            d, t = ggroups[group_name]
            tris = []
            for ent in gmsh.model.getEntitiesForPhysicalGroup(d, t):
                ets, ens = gmsh.model.mesh.getElementsByType(2, ent)[0:2] if False else (None, None)
            # collect triangle node-sets across the group's surfaces
            node_tris = []
            for ent in gmsh.model.getEntitiesForPhysicalGroup(d, t):
                etypes, etags, enodes = gmsh.model.mesh.getElements(2, ent)
                for et, en in zip(etypes, enodes):
                    if et == 2:  # 3-node triangle
                        arr = list(en)
                        for i in range(0, len(arr), 3):
                            node_tris.append(tuple(arr[i:i + 3]))
            # union-find over shared nodes
            parent = {}
            def find(x):
                parent.setdefault(x, x)
                while parent[x] != x:
                    parent[x] = parent[parent[x]]; x = parent[x]
                return x
            def union(a_, b_):
                parent[find(a_)] = find(b_)
            for tri in node_tris:
                for nd in tri:
                    union(("n", nd), ("t", id(tri)))
            roots = {find(("n", nd)) for tri in node_tris for nd in tri}
            return len(roots), len(node_tris)
        for gn in ("TopMetal2", "TopMetal1", "Metal1"):
            r = n_components(gn)
            if r:
                print(f"  connectivity[{gn}]: {r[0]} component(s), {r[1]} triangles")
        print("mesh-only: stopping before solve")
        return

    prob = rf.Problem(g)
    result = prob.sweep(np.asarray(a.freqs))
    print(f"solve:  {prob.n_tets} tets, {prob.n_dofs} DOFs")
    result.to_touchstone(str(out), z0=50.0)
    print(f"wrote:  {out}")

    # quick L,Q readback
    z0 = 50.0
    nport = result.sparams[0].shape[0]
    print(f"\n{'f[GHz]':>8} {'L[nH]':>9} {'Q':>8}")
    for k, f in enumerate(a.freqs):
        S = result.sparams[k]
        w = 2 * np.pi * f
        if nport == 1:
            # 1-port differential: Zin = z0(1+S)/(1-S); L = Im(Zin)/w directly.
            Zin = z0 * (1 + S[0, 0]) / (1 - S[0, 0])
            L = Zin.imag / w
            Q = Zin.imag / Zin.real if Zin.real else float("nan")
        else:
            # 2-port single-ended Niknejad (matches analyze_lhs)
            I = np.eye(nport)
            Z = np.sqrt(z0) * (I + S) @ np.linalg.inv(I - S) * np.sqrt(z0)
            Y = np.linalg.inv(Z)
            L = -1.0 / (w * Y[0, 0].imag) if Y[0, 0].imag else float("nan")
            Q = -Y[0, 0].imag / Y[0, 0].real if Y[0, 0].real else float("nan")
        print(f"{f/1e9:>8.3f} {L*1e9:>9.3f} {Q:>8.2f}")


if __name__ == "__main__":
    main()
