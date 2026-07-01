#!/usr/bin/env python3
"""
generate_sweep_palace.py
========================
Build a SWEEP of symmetric octagonal inductor Palace models for a fixed-frequency
predicted-vs-EM comparison (Hershenson et al. Fig. 5 style).

For each target inductance L = 1..N_MAX nH, the GP is solved at integer turns
(design_inductor(..., integer_turns=True), the default: hits the target L exactly
with an integer N -- see PROJECT_OVERVIEW.md (sec. 2)), the coil is drawn with the
KlayoutDrawing flow, and a
Palace FEM model is created with gds2palace. NOTHING is simulated here -- the model
dirs are meant to be run on CLEPS as a SLURM array job (cleps/run_sweep_array.sh).

Frequency: GP operating point and Palace fpoint are both 2.5 GHz. The inductance
coefficients are the SG13G2 symmetric AC fit at 2.4 GHz (closest available; the 0.1
GHz gap is negligible for L). A manifest CSV records every GP prediction + model dir.

Run from the repo root:
    python3 scripts/generate_sweep_palace.py            # L = 1..20 nH
    python3 scripts/generate_sweep_palace.py --max 12   # L = 1..12 nH
"""
import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import klayout.db as db
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "projects" / "inductor_lab" / "src"))
import gds2palace as gp
from inductor_lab.gp.spiral import (
    InductanceCoefficients, Topology, design_inductor)
from inductor_lab.pdk.sg13g2 import BACKLAPPING, SG13G2

FREQ = 2.5e9                       # operating frequency for GP + Palace
# Frequency-sweep mode (--fsweep): adaptive Linear sweep so we can extract L below
# self-resonance and locate the actual SRF, instead of a single corrupted 2.5 GHz point.
FSWEEP = False
FSWEEP_START, FSWEEP_STOP, FSWEEP_STEP = 0.1e9, 10.0e9, 0.1e9
SWEEP_DIR = REPO / "scripts" / ".out" / "sweep_2p5ghz"          # set in main() per mode

# Grounded-substrate fixture (--grounded). The default fixture floats the substrate: the
# only reference is the Metal1 frame, and the outer box is absorbing on all 6 faces, so the
# 180 um substrate has no ground return (confirmed: measured C_p ~= C_ox, and the Yue k7
# shunt term -- which assumes a grounded backside -- disagrees). SG13G2's real reference IS a
# grounded, back-metallised wafer (the PDK stackup defines SUBGND + BACKSIDEGND as PEC-like
# LOWLOSS planes). When True, build_one adds: (1) a BACKSIDEGND (251) plane under the whole
# footprint, and (2) a guard-ring substrate contact (Activ 1 + Cont 6) under the Metal1 frame,
# so frame -> epi/substrate -> backside form ONE ground net referenced by the ports. No plane
# under the coil (that would shield it) -- just bulk substrate grounded at the backside.
GROUNDED_BACKSIDE = False

# Differential-drive fixture (--diff). Erases the ground frame entirely and drives the coil
# with a SINGLE differential in-plane lumped port bridging the two TopMetal2 leads (field in x).
# The result is a 1-port whose input impedance IS the coil's differential impedance
# (Z_diff = Im(Zin)/w, Q = Im/Re) -- how a symmetric center-tapped coil is actually used --
# with no ground-frame / common-mode reference. Mutually exclusive with the frame-based options
# above (no frame => nothing for GROUNDED_BACKSIDE's guard ring to tie to).
DIFFERENTIAL_DRIVE = False

# Backside-ground fixture (--backside). Mirrors simulate_rapidfem.py --port backside EXACTLY:
# erases the Metal1 guard ring and instead references the two ports to an explicit backside
# ground plane. There is NO frame and NO guard-ring substrate contact. When True, build_one:
#   (1) draws a BACKSIDEGND (251) plane under the whole footprint,
#   (2) draws a small LOCAL Metal1 (8) ground pad directly under each pin (the port's lower
#       reference -- the 201/202 ports stay short & valid: Metal1 -> TopMetal2),
#   (3) draws a GNDPLUG (252, new LOWLOSS layer spanning Metal1-top down to BACKSIDEGND-top)
#       column co-located with each ground pad, so pad -> plug -> backside plane is ONE PEC net
#       -- the Palace analogue of rapidfem's grounded through-substrate plug.
# Mutually exclusive with --grounded/--diff (both assume the frame). The long direct
# BACKSIDEGND->TopMetal2 lumped port is deliberately avoided (invalid distributed feed, per
# the rapidfem note); the short port + plug reproduces the same ground topology.
BACKSIDE_NOFRAME = False

# Lateral domain padding for the --backside fixture. The BACKSIDEGND plane (and hence the
# gds2palace substrate/air domain, which grows to metal_bbox + margin) is sized to
# coil_bbox * BACKSIDE_LATERAL_PAD, centred -- matching simulate_rapidfem.py's --pad so the two
# solvers see the SAME lateral truncation. rapidfem is run with --pad 3.0.
BACKSIDE_LATERAL_PAD = 3.0

# -- SG13G2 symmetric AC (2.4 GHz) coefficients -------------------------------
_ac_path = REPO / "ASITIC" / "coefficients_sym_sg13g2_2.4ghz.py"
_spec = importlib.util.spec_from_file_location("ac", _ac_path)
_ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ac)
COEFFS = InductanceCoefficients(_ac.BETA, _ac.A1, _ac.A2, _ac.A3, _ac.A4, _ac.A5)
PDK = replace(SG13G2, t_sub=BACKLAPPING[200])

# -- KlayoutDrawing grid ------------------------------------------------------
# The only thing the drawing needs from the SG13G2 klayout PDK is the drawing grid
# (0.005 um). Hardcode it so this generator runs anywhere -- e.g. directly on CLEPS --
# without shipping the 229 MB klayout PDK tech tree. `tech` is kept as a None sentinel
# so the roundToGrid(tech, ...) call sites stay unchanged.
GRID = 0.005
tech = None


def roundToGrid(tech, num, floor=False, ceil=False):
    grid = GRID
    if floor or ceil:
        return np.floor(num / grid) * grid if floor else np.ceil(num / grid) * grid
    return np.round(num / grid) * grid


def roundPointToGrid(tech, x, y):
    return roundToGrid(tech, x), roundToGrid(tech, y)


def octSegment(w, r, e):
    a = roundToGrid(tech, (2 * r) / (np.sqrt(2) + 2))
    r = roundToGrid(tech, r)
    c = roundToGrid(tech, (2 * r) / (np.sqrt(2) + 2) + w / (np.sqrt(2) + 1), floor=True)
    x, y = r, e / 2
    p1 = db.DPoint(x, y); y = r - a
    p2 = db.DPoint(x, y); x -= a; y += a
    p3 = db.DPoint(x, y); x = e / 2
    p4 = db.DPoint(x, y); y -= w
    p5 = db.DPoint(x, y); x = r - c
    p6 = db.DPoint(x, y); x = r - w; y = r - c
    p7 = db.DPoint(x, y); y = e / 2
    p8 = db.DPoint(x, y)
    return db.DPolygon([p1, p2, p3, p4, p5, p6, p7, p8])


def make45Bridge(w, l, s, addVias=False):
    x, y = -(2 * w + s) / 2, -l / 2
    p1 = db.DPoint(*roundPointToGrid(tech, x, y)); x += w
    p2 = db.DPoint(*roundPointToGrid(tech, x, y)); y += s / (1 + np.sqrt(2))
    p3 = db.DPoint(*roundPointToGrid(tech, x, y)); x += w + s; y += w + s
    p4 = db.DPoint(x, y); y += (w + s) / (1 + np.sqrt(2))
    p5 = db.DPoint(*roundPointToGrid(tech, x, y)); x -= w
    p6 = db.DPoint(*roundPointToGrid(tech, x, y)); y -= s / (1 + np.sqrt(2))
    p7 = db.DPoint(*roundPointToGrid(tech, x, y)); x -= w + s; y -= w + s
    p8 = db.DPoint(*roundPointToGrid(tech, x, y))
    if addVias:
        p1.y -= w; p2.y -= w; p5.y += w; p6.y += w
    return db.DPolygon([p1, p2, p3, p4, p5, p6, p7, p8])


def build_one(sol, out_dir, geom_only=False):
    """Draw `sol` and build its Palace model dir. Returns (ind_name, sim_path).

    geom_only=True: write ONLY the drawn *_forEM.gds (coil + underpass + vias +
    201/202 port markers + ground frame -- everything simulate_rapidfem needs) and
    skip the heavy Palace gmsh/create_palace build. Used by the rapidfem LHS, which
    does its own meshing. Each coil gets its OWN dir (out_dir/<ind_name>/) so the
    fit's best_s2p(model_dir) later resolves to exactly that coil's rapidfem .s2p."""
    w = roundToGrid(tech, sol.w * 1e6)
    s = roundToGrid(tech, sol.s * 1e6)
    r = roundToGrid(tech, sol.d_out / 2 * 1e6)
    N = round(sol.n)
    conn_len = 3 * w
    via_width = SG13G2.top_vias[0].width * 1e6
    via_spacing = SG13G2.top_vias[0].space * 1e6
    via_enc = SG13G2.top_vias[0].enc_upper * 1e6
    via_pitch = via_width + via_spacing
    e = roundToGrid(tech, w + s + (w + 2 * s) / (1 + np.sqrt(2)))

    layout = db.Layout()
    layout.technology_name = "sg13g2"
    top = layout.create_cell("TOP")
    inductor = layout.create_cell("Inductor")
    signal_layer = layout.layer(134, 0)
    underpass_layer = layout.layer(126, 0)
    signal_pin_layer = layout.layer(134, 2)
    signal_underpass_via_layer = layout.layer(133, 0)

    for n in range(N):
        seg = octSegment(w, r - n * (w + s), e)
        for i in range(4):
            seg.transform(db.DTrans(rot=45 * i))
            inductor.shapes(signal_layer).insert(seg)

    via_cell = layout.create_cell("VIA")
    via_cell.shapes(signal_underpass_via_layer).insert(db.DBox(0, 0, via_width, via_width))
    for n in range(N - 1):
        bridge = make45Bridge(w, e, s, addVias=True)
        bridge.transform(db.DTrans(rot=45))
        bridge.transform(db.DTrans(0, (-1) ** (2 + n) * (r - (1 + n) * (w + s) + s / 2)))
        upass = bridge.transformed(db.DTrans(rot=90, mirrx=True))
        inductor.shapes(signal_layer).insert(bridge)
        inductor.shapes(underpass_layer).insert(upass)
        left_bottom = min(upass.each_point_hull(), key=lambda p: (p.x, p.y))
        right_bottom = max(upass.each_point_hull(), key=lambda p: (p.x, -p.y))
        via_number = np.floor((w - 2 * via_enc + via_spacing) / via_pitch)
        array_size = via_number * via_width + (via_number - 1) * via_spacing
        centering_offset = via_enc + (w - 2 * via_enc - array_size) / 2
        inductor.insert(db.DCellInstArray(via_cell.cell_index(),
            db.DTrans(left_bottom.x + centering_offset, left_bottom.y + centering_offset),
            db.DVector(via_pitch, 0), db.DVector(0, via_pitch), via_number, via_number))
        inductor.insert(db.DCellInstArray(via_cell.cell_index(),
            db.DTrans(right_bottom.x - centering_offset - via_width, right_bottom.y + centering_offset),
            db.DVector(-via_pitch, 0), db.DVector(0, via_pitch), via_number, via_number))

    for n in range(N):
        patch = db.DBox(db.DPoint(r, e / 2), db.DPoint(r - w, -e / 2)); patch.move(-n * (w + s), 0)
        inductor.shapes(signal_layer).insert(patch)
        inductor.shapes(signal_layer).insert(patch.transformed(db.DTrans(rot=90)))
    patch = db.DBox(db.DPoint(e / 2, r - (N - 1) * (w + s)), db.DPoint(-e / 2, r - w - (N - 1) * (w + s)))
    if not (N % 2):
        patch = patch.transformed(db.DTrans(rot=90))
    inductor.shapes(signal_layer).insert(patch)

    conn = db.DBox(db.DPoint(-e / 2 - w / 2, -r + w), db.DPoint(-e / 2 + w / 2, -r - conn_len))
    inductor.shapes(signal_layer).insert(conn)
    inductor.shapes(signal_layer).insert(conn.moved(e, 0))

    region = db.Region(inductor.begin_shapes_rec(signal_layer)); region.merge()
    inductor.shapes(signal_layer).clear()
    inductor.shapes(signal_layer).insert(region)

    pin_height = s
    pin = db.DBox(db.DPoint(-e / 2 - w / 2, -r - conn_len + pin_height), db.DPoint(-e / 2 + w / 2, -r - conn_len))
    inductor.shapes(signal_pin_layer).insert(pin)
    inductor.shapes(signal_pin_layer).insert(pin.moved(e, 0))

    ind_name = f"sweep_L{int(round(sol.L*1e9))}_N{N}_{w}_{s}_{2*r}".replace(".", "p")

    palace_pin1_layer = layout.layer(201, 0)
    palace_pin2_layer = layout.layer(202, 0)
    inductor.shapes(palace_pin1_layer).insert(pin.enlarged(0, -pin.height() / 4).moved(0, -pin.height() / 4))
    inductor.shapes(palace_pin2_layer).insert(pin.enlarged(0, -pin.height() / 4).moved(e, -pin.height() / 4))
    ind_bbox = inductor.dbbox()

    if DIFFERENTIAL_DRIVE:
        # ONE differential in-plane port (layer 203) bridging the two TopMetal2 leads across
        # the gap between them (field in x), and NO ground frame. Measures the coil's
        # DIFFERENTIAL impedance directly as a 1-port (Z_diff = Im(Zin)/w) -- the way a
        # symmetric center-tapped coil is actually driven -- instead of two single-ended ports
        # against a ground reference. Mirrors simulate_rapidfem.py port_mode="diff": the plate
        # spans inner-edge to inner-edge of the two feedlines, width >= gap, on the stubs.
        gl, gr = -e / 2 + w / 2, e / 2 - w / 2                 # inner edges of the two leads
        py_lo = -r - conn_len                                  # bottom of the feedline stubs
        py_hi = py_lo + min(conn_len, max(e - w, 20.0))        # width >= gap, stays on the leads
        inductor.shapes(layout.layer(203, 0)).insert(
            db.DBox(db.DPoint(gl, py_lo), db.DPoint(gr, py_hi)))
    elif BACKSIDE_NOFRAME:
        # NO frame. Reference the two 201/202 ports to an explicit backside ground plane,
        # reached from a local Metal1 ground pad under each pin via a through-substrate PEC
        # plug. Mirrors simulate_rapidfem.py --port backside (local ground pad + ground plug +
        # BACKSIDEGND plane, all one PEC net; the short Metal1->TopMetal2 port stays valid).
        gnd_metal_layer = layout.layer(8, 0)         # Metal1 local ground pads
        plug_layer = layout.layer(252, 0)            # GNDPLUG column (Metal1-top -> backside-top)
        bsg_layer = layout.layer(251, 0)             # BACKSIDEGND plane
        plug_thick = roundToGrid(tech, 2.0)          # y-thickness of the thin plug strip [um]
        for xoff in (0.0, e):
            p = pin.moved(xoff, 0)
            # Local Metal1 ground pad covering the 201/202 marker so the port's lower edge welds.
            inductor.shapes(gnd_metal_layer).insert(p.enlarged(w, w))
            # Ground plug straight down to the backside plane. Mirrors simulate_rapidfem.py
            # --port backside EXACTLY: a THIN pin-width sheet (rapidfem g.plate: pw wide in x,
            # ~0 in y), NOT a fat block. rapidfem does NOT carve any air/oxide around it -- the
            # plug sits directly in the conductive Substrate/EPI and that coupling IS the
            # intended C_si. Kept a thin y-sliver so the solid-volume substrate contact is minimal.
            cy = p.center().y
            inductor.shapes(plug_layer).insert(
                db.DBox(db.DPoint(p.left, cy - plug_thick / 2),
                        db.DPoint(p.right, cy + plug_thick / 2)))
        # BACKSIDEGND plane sized to rapidfem's footprint (coil_bbox * BACKSIDE_LATERAL_PAD,
        # centred): gds2palace grows the substrate/air to this plane + margin, so the whole
        # Palace lateral domain tracks rapidfem --pad instead of the tight ~100 um default gap.
        hx = roundToGrid(tech, max(abs(ind_bbox.left), abs(ind_bbox.right)) * BACKSIDE_LATERAL_PAD)
        hy = roundToGrid(tech, max(abs(ind_bbox.bottom), abs(ind_bbox.top)) * BACKSIDE_LATERAL_PAD)
        inductor.shapes(bsg_layer).insert(db.DBox(-hx, -hy, hx, hy))
    else:
        palace_ground_layer = layout.layer(8, 0)
        # Ground frame matched to the gds2palace example (synthesize_ihp_inductor_v1.py):
        # gap = D/2 = r (was r/2 -> twice as far out, so much less frame capacitance that
        # was dragging the EM self-resonance onto the operating frequency), width min(20,5w).
        frame_margin = r
        frame_width = min(20.0, 5.0 * w)
        gframe = db.DPolygon(ind_bbox.enlarged(roundToGrid(tech, frame_margin + frame_width)))
        gframe_hole = ind_bbox.enlarged(roundToGrid(tech, frame_margin))
        gframe.insert_hole(gframe_hole)
        gframe_pin = db.DBox(db.DPoint(pin.left - 5, pin.top), db.DPoint(pin.right + e + 5, gframe_hole.bottom))
        inductor.shapes(palace_ground_layer).insert(gframe)
        inductor.shapes(palace_ground_layer).insert(gframe_pin)
        greg = db.Region(inductor.begin_shapes_rec(palace_ground_layer)); greg.merge()
        inductor.shapes(palace_ground_layer).clear()
        inductor.shapes(palace_ground_layer).insert(greg)

        if GROUNDED_BACKSIDE:
            # (1) BACKSIDEGND (251) plane under the WHOLE footprint -> grounds the substrate
            # backside. Sized to the full frame+pin bbox +30 um (stays inside the substrate
            # dielectric box, which is metal_bbox + margin(50)).
            # (2) Guard-ring substrate contact: Activ (1) + Cont (6) under the Metal1 frame ->
            # galvanically ties the frame down to the epi/substrate top (Activ z0-0.4 sits on the
            # semiconductor). The conductive substrate (sigma=2) then carries ground to
            # BACKSIDEGND, so frame + substrate + backside are ONE net referenced by the ports.
            full_bbox = inductor.dbbox()
            inductor.shapes(layout.layer(251, 0)).insert(
                db.DBox(full_bbox.enlarged(roundToGrid(tech, 30.0))))
            for lnum in (1, 6):                          # Activ, Cont (below Metal1 frame)
                inductor.shapes(layout.layer(lnum, 0)).insert(greg)

    top.clear()
    top.insert(db.CellInstArray(inductor.cell_index(), db.Trans()))

    if geom_only:
        coil_dir = out_dir / ind_name
        coil_dir.mkdir(parents=True, exist_ok=True)
        layout.write(str(coil_dir / f"{ind_name}_forEM.gds"))
        return ind_name, str(coil_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    em_gds = str(out_dir / f"{ind_name}_forEM.gds")
    layout.write(em_gds)

    XML_filename = str(REPO / "resources" / "SG13G2_200um.xml")
    if FSWEEP:
        freq_settings = {"fstart": FSWEEP_START, "fstop": FSWEEP_STOP, "fstep": FSWEEP_STEP,
                         "fpoint": [FREQ], "adaptive_sweep": True}
    else:
        freq_settings = {"fpoint": [FREQ], "adaptive_sweep": False}
    settings = {
        "unit": 1e-6, "margin": 50, **freq_settings,
        "refined_cellsize": 5, "cells_per_wavelength": 10,
        "meshsize_max": 70, "adaptive_mesh_iterations": 0,
        "order": 2, "no_gui": True, "no_preview": True,
        "merge_polygon_size": 1.5, "preprocess_gds": True,
    }
    sim_path = gp.utilities.create_sim_path(str(out_dir), ind_name)
    simulation_ports = gp.simulation_setup.all_simulation_ports()
    if DIFFERENTIAL_DRIVE:
        # single differential in-plane port across the two leads (layer 203, field in x)
        simulation_ports.add_port(gp.simulation_setup.simulation_port(
            portnumber=1, voltage=1, port_Z0=50, source_layernum=203,
            target_layername="TopMetal2", direction="x"))
    else:
        simulation_ports.add_port(gp.simulation_setup.simulation_port(
            portnumber=1, voltage=1, port_Z0=50, source_layernum=201,
            from_layername="Metal1", to_layername="TopMetal2", direction="z"))
        simulation_ports.add_port(gp.simulation_setup.simulation_port(
            portnumber=2, voltage=1, port_Z0=50, source_layernum=202,
            from_layername="Metal1", to_layername="TopMetal2", direction="z"))
    materials_list, dielectrics_list, metals_list = gp.stackup_reader.read_substrate(XML_filename)
    layernumbers = metals_list.getlayernumbers()
    layernumbers.extend(simulation_ports.portlayers)
    allpolygons = gp.gds_reader.read_gds(em_gds, layernumbers, purposelist=[0],
                                         metals_list=metals_list, preprocess=settings["preprocess_gds"],
                                         merge_polygon_size=settings["merge_polygon_size"],
                                         cellname="Inductor")
    settings.update({"simulation_ports": simulation_ports, "materials_list": materials_list,
                     "dielectrics_list": dielectrics_list, "metals_list": metals_list,
                     "layernumbers": layernumbers, "allpolygons": allpolygons,
                     "sim_path": sim_path, "model_basename": ind_name})
    excite_ports = simulation_ports.all_active_excitations()
    config_name, data_dir = gp.simulation_setup.create_palace(excite_ports, settings)
    gp.utilities.create_run_script(sim_path)
    with open(config_name) as f:
        cfg = json.load(f)
    cfg["Solver"]["Linear"]["MaxIts"] = 1000
    with open(config_name, "w") as f:
        json.dump(cfg, f, indent=4)
    return ind_name, sim_path


def main():
    global FSWEEP, SWEEP_DIR, GROUNDED_BACKSIDE, DIFFERENTIAL_DRIVE, BACKSIDE_NOFRAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=20, help="max target inductance [nH]")
    ap.add_argument("--min", type=int, default=1, help="min target inductance [nH]")
    ap.add_argument("--fsweep", action="store_true",
                    help=f"adaptive frequency sweep {FSWEEP_START/1e9:g}-{FSWEEP_STOP/1e9:g} GHz "
                         "(isolated output dir) instead of a single 2.5 GHz point")
    ap.add_argument("--grounded", action="store_true",
                    help="grounded-substrate fixture: add BACKSIDEGND plane + guard-ring "
                         "substrate contact under the frame (isolated output dir)")
    ap.add_argument("--diff", action="store_true",
                    help="differential drive: erase the ground frame, use one differential "
                         "in-plane port across the two leads (1-port Z_diff; isolated dir)")
    ap.add_argument("--backside", action="store_true",
                    help="backside-ground fixture: erase the frame, reference the two ports to "
                         "an explicit BACKSIDEGND plane via a through-substrate plug (mirrors "
                         "simulate_rapidfem --port backside; isolated output dir)")
    a = ap.parse_args()
    FSWEEP = a.fsweep
    GROUNDED_BACKSIDE = a.grounded
    DIFFERENTIAL_DRIVE = a.diff
    BACKSIDE_NOFRAME = a.backside
    if sum((DIFFERENTIAL_DRIVE, GROUNDED_BACKSIDE, BACKSIDE_NOFRAME)) > 1:
        ap.error("--diff, --grounded and --backside are mutually exclusive")
    if FSWEEP:
        SWEEP_DIR = REPO / "scripts" / ".out" / "sweep_2p5ghz_fsweep"
    if GROUNDED_BACKSIDE:
        SWEEP_DIR = SWEEP_DIR.with_name(SWEEP_DIR.name + "_gnd")
    if DIFFERENTIAL_DRIVE:
        SWEEP_DIR = SWEEP_DIR.with_name(SWEEP_DIR.name + "_diff")
    if BACKSIDE_NOFRAME:
        SWEEP_DIR = SWEEP_DIR.with_name(SWEEP_DIR.name + "_bsg")
    tag = ("  [GROUNDED substrate]" if GROUNDED_BACKSIDE else
           "  [DIFFERENTIAL drive, no frame]" if DIFFERENTIAL_DRIVE else
           "  [BACKSIDE ground, no frame]" if BACKSIDE_NOFRAME else "")
    print(f"Mode: {'FREQUENCY SWEEP %g-%g GHz' % (FSWEEP_START/1e9, FSWEEP_STOP/1e9) if FSWEEP else 'single 2.5 GHz point'}"
          f"{tag}  ->  {SWEEP_DIR.relative_to(REPO)}")

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for Lt in range(a.min, a.max + 1):
        sol = design_inductor(Lt, FREQ, PDK, topology=Topology.DIFFERENTIAL,
                              coeffs=COEFFS)   # integer_turns=True by default
        ind_name, sim_path = build_one(sol, SWEEP_DIR)
        model_dir = str(Path(sim_path).resolve().relative_to(REPO))
        rows.append(dict(idx=len(rows) + 1, target_nH=Lt, N=round(sol.n),
                         w_um=round(sol.w * 1e6, 4), s_um=round(sol.s * 1e6, 4),
                         d_out_um=round(sol.d_out * 1e6, 3), gp_L_nH=round(sol.L * 1e9, 4),
                         gp_Q=round(sol.Q, 4), gp_fsr_GHz=round(sol.f_sr / 1e9, 4),
                         model_dir=model_dir))
        print(f"[{len(rows):2d}] L={Lt:2d}nH  N={round(sol.n)}  "
              f"w={sol.w*1e6:6.2f} d_out={sol.d_out*1e6:6.1f}  Q={sol.Q:5.2f}  -> {ind_name}")

    manifest = SWEEP_DIR / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        # lineterminator="\n": avoid the csv module's default CRLF, which leaves a
        # trailing \r on awk-parsed fields in cleps/run_sweep_array.sh.
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nWrote {len(rows)} models + manifest: {manifest}")
    print("Next: copy scripts/.out/sweep_2p5ghz/ to CLEPS and "
          "`sbatch cleps/run_sweep_array.sh`")


if __name__ == "__main__":
    main()
