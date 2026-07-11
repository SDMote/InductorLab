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
import multiprocessing as mp
import sys
from dataclasses import replace
from pathlib import Path

import klayout.db as db
import numpy as np

REPO = Path(__file__).resolve().parents[2]   # scripts/palace/<file> -> repo root
sys.path.insert(0, str(REPO / "projects" / "inductor_lab" / "src"))
import gds2palace as gp
from inductor_lab.gp.spiral import InductanceCoefficients
from inductor_lab.pdk.sg13g2 import BACKLAPPING, SG13G2

# Per-coil gmsh build timeout: a hung mesh (uncatchable C-level loop) is killed and skipped.
PER_MODEL_TIMEOUT = 480     # s (legit builds ~1-8 min)

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

# Mark the BACKSIDEGND plane as a true PEC GROUND (0 V reference) instead of the default
# floating finite-sigma (Conductivity) sheet. gds2palace emits every drawn conductor as a
# surface-impedance "Conductivity" boundary whose potential floats; setting this True
# post-patches the config to move the LOWLOSS (sigma=1e10) backside sheet into a "PEC"
# boundary, so it becomes a fixed-0 V current sink / shunt-C reference. Only meaningful with
# a backside plane present (BACKSIDE_NOFRAME); intended for the differential no-plug fixture.
BACKSIDE_PEC_GROUND = False

# Control test: zero out the Substrate/EPI conductivity (make the Si a LOSSLESS dielectric,
# eps=11.9, sigma->0) so we can compare Re(Zin) with vs without substrate conduction on the
# SAME mesh. If Q/Re(Zin) are unchanged, Palace was not dissipating in the Si; if they differ,
# the substrate loss is real (and the Yue/GP formula over-predicts it). Patches the config
# post-build. Sigma does not affect meshing, so the sigma=2 and sigma=0 meshes are identical.
LOSSLESS_SUBSTRATE = False

# Lateral domain padding for the --backside fixture. The BACKSIDEGND plane (and hence the
# gds2palace substrate/air domain, which grows to metal_bbox + margin) is sized to
# coil_bbox * BACKSIDE_LATERAL_PAD, centred -- matching simulate_rapidfem.py's --pad so the two
# solvers see the SAME lateral truncation. rapidfem is run with --pad 3.0.
BACKSIDE_LATERAL_PAD = 3.0

# -- SG13G2 symmetric AC (2.4 GHz) coefficients -------------------------------
_ac_path = REPO / "ASITIC" / "coefficients_sym_sg13g2.py"
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
        if BACKSIDE_NOFRAME:
            # Differential drive OVER a grounded backside plane, but with NO through-substrate
            # plug: the only coil->backside path is the lossy Substrate/EPI (the physical R_p),
            # not the artificial low-R PEC plug. The 203 port references the two leads to each
            # other; the BACKSIDEGND plane grounds the substrate backside so the substrate loss
            # is real. Plane sized like the --backside fixture (coil_bbox * BACKSIDE_LATERAL_PAD).
            hx = roundToGrid(tech, max(abs(ind_bbox.left), abs(ind_bbox.right)) * BACKSIDE_LATERAL_PAD)
            hy = roundToGrid(tech, max(abs(ind_bbox.bottom), abs(ind_bbox.top)) * BACKSIDE_LATERAL_PAD)
            inductor.shapes(layout.layer(251, 0)).insert(db.DBox(-hx, -hy, hx, hy))
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
    if BACKSIDE_PEC_GROUND:
        # Reclassify the LOWLOSS (sigma=1e10) backside sheet from a floating finite-sigma
        # Conductivity boundary to a PEC ground (fixed 0 V reference / current sink).
        bnd = cfg["Boundaries"]
        conds = bnd.get("Conductivity", [])
        gnd_attrs, kept = [], []
        for c in conds:
            sig = c.get("Conductivity")
            sig = max(sig) if isinstance(sig, (list, tuple)) else sig
            (gnd_attrs.extend(c.get("Attributes", [])) if sig and sig >= 1e9 else kept.append(c))
        if gnd_attrs:
            bnd["Conductivity"] = kept
            if not kept:
                bnd.pop("Conductivity")
            pec = bnd.setdefault("PEC", {"Attributes": []})
            pec["Attributes"] = sorted(set(pec.get("Attributes", []) + gnd_attrs))
            print(f"BACKSIDE_PEC_GROUND: moved attrs {gnd_attrs} -> PEC ground")
    if LOSSLESS_SUBSTRATE:
        # Zero the Si conduction: substrate/EPI become lossless dielectric (keep eps=11.9).
        n_zeroed = 0
        for m in cfg["Domains"]["Materials"]:
            sig = m.get("Conductivity")
            smax = max(sig) if isinstance(sig, (list, tuple)) else sig
            if smax in (2.0, 5.0):            # Substrate (2) / EPI (5)
                m.pop("Conductivity"); m["LossTan"] = 0.0; n_zeroed += 1
        print(f"LOSSLESS_SUBSTRATE: zeroed conductivity on {n_zeroed} Si domain(s)")
    with open(config_name, "w") as f:
        json.dump(cfg, f, indent=4)
    return ind_name, sim_path


def _build_worker(sol, out_dir, q, geom_only=False):
    try:
        ind_name, sim_path = build_one(sol, out_dir, geom_only=geom_only)
        q.put(("ok", ind_name, sim_path))
    except Exception as e:
        q.put(("error", str(e)[:55], ""))


def build_with_timeout(sol, out_dir, timeout=PER_MODEL_TIMEOUT, geom_only=False):
    """Build one model in a SUBPROCESS so a gmsh HANG (a C-level infinite loop no try/except
    can catch) is killable. Returns (ind_name, sim_path) on success, else (None, reason). A
    killed model's half-built dir has no config.json, so the solve array ignores it."""
    q = mp.Queue()
    proc = mp.Process(target=_build_worker, args=(sol, out_dir, q, geom_only))
    proc.start(); proc.join(timeout)
    if proc.is_alive():
        proc.terminate(); proc.join()
        return None, "TIMEOUT (gmsh hang)"
    try:
        status, a, b = q.get_nowait()
    except Exception:
        return None, "no result (crashed)"
    return (a, b) if status == "ok" else (None, a)


def _write_manifest(out_dir, rows):
    # temp file + atomic rename: a SLURM kill mid-write can never leave a half-written manifest.
    if not rows:
        return
    tmp = out_dir / "manifest.csv.tmp"
    with open(tmp, "w", newline="") as f:      # lineterminator="\n": no CRLF for awk in the array
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        wtr.writeheader(); wtr.writerows(rows)
    tmp.replace(out_dir / "manifest.csv")


def main():
    """THE inductor sweep generator: GP-design L=min..max nH, build each in the chosen Palace
    fixture. Uses spiral_cturn.design (full Mohan d_out form + fringing C_p) with the canonical
    coefficients_sym_sg13g2.py. Replaces the old generate_sweep{,_rfcoeffs}.py."""
    global FSWEEP, FSWEEP_STOP, FREQ
    global GROUNDED_BACKSIDE, DIFFERENTIAL_DRIVE, BACKSIDE_NOFRAME, BACKSIDE_PEC_GROUND, LOSSLESS_SUBSTRATE
    from types import SimpleNamespace
    import spiral_cturn as sct                   # local: keeps this module light for build-only importers
    ap = argparse.ArgumentParser(description="Design (GP) + build the Palace inductor sweep.")
    ap.add_argument("--min", type=int, default=1, help="min target inductance [nH]")
    ap.add_argument("--max", type=int, default=20, help="max target inductance [nH]")
    ap.add_argument("--freq", type=float, default=2.5, help="operating frequency [GHz]")
    ap.add_argument("--srf", type=float, default=7.0, help="SRF floor [GHz]; set = --freq to release it")
    ap.add_argument("--w-max", type=float, default=28.0, help="max trace width [um] (L-fit box cap)")
    ap.add_argument("--s-max", type=float, default=18.0, help="max spacing [um] (L-fit box cap)")
    ap.add_argument("--out", default="sweep", help="output dir under scripts/.out/")
    ap.add_argument("--geom-only", action="store_true", help="write only *_forEM.gds (no Palace mesh)")
    # -- fixture (default: ground frame; pick ONE of the below) --
    ap.add_argument("--diffpec", action="store_true",
                    help="DIFFERENTIAL 1-port over a PEC-ground backside plane, NO plug (the true "
                         "backside ground the L coeffs are fit to).")
    ap.add_argument("--backside", action="store_true",
                    help="backside plug 2-port (BACKSIDEGND plane + through-substrate plug).")
    ap.add_argument("--grounded", action="store_true", help="grounded-substrate frame fixture.")
    ap.add_argument("--losslesssub", action="store_true",
                    help="control: zero the Si conductivity (does Palace dissipate in the substrate?).")
    a = ap.parse_args()
    if sum((a.diffpec, a.backside, a.grounded)) > 1:
        ap.error("--diffpec, --backside, --grounded are mutually exclusive")

    FREQ = a.freq * 1e9; sct.FREQ = FREQ
    FSWEEP = True; FSWEEP_STOP = 15.0e9
    BACKSIDE_NOFRAME = a.diffpec or a.backside      # both keep the BACKSIDEGND plane
    DIFFERENTIAL_DRIVE = a.diffpec                  # diffpec => single differential 1-port, no plug
    BACKSIDE_PEC_GROUND = a.diffpec                 # diffpec => plane is a true PEC ground
    GROUNDED_BACKSIDE = a.grounded
    LOSSLESS_SUBSTRATE = a.losslesssub

    OUT = REPO / "scripts" / ".out" / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    fixture = ("diffpec (differential PEC-ground, no plug)" if a.diffpec else
               "backside plug 2-port" if a.backside else
               "grounded frame" if a.grounded else "frame")
    print(f"Sweep L={a.min}..{a.max} nH @ {a.freq} GHz | SRF>={a.srf} | fixture={fixture}"
          f"{' | LOSSLESS Si' if a.losslesssub else ''}\n  coeffs: {sct.COEFFS}\n  -> {OUT.relative_to(REPO)}\n")

    rows, skipped = [], []
    for L in range(a.min, a.max + 1):
        d = sct.best_int(L, _design=sct.design, min_srf_hz=a.srf * 1e9,
                         w_max=a.w_max * 1e-6, s_max=a.s_max * 1e-6)
        if d is None:
            skipped.append(f"{L}(infeasible)")
            print(f"  L={L:2d}  INFEASIBLE (SRF>={a.srf} GHz within w<={a.w_max}, s<={a.s_max})")
            continue
        N = round(d["n"])
        sol = SimpleNamespace(n=N, w=d["w"], s=d["s"], d_out=d["d_out"], L=d["L"])
        ind_name, sim_path = build_with_timeout(sol, OUT, PER_MODEL_TIMEOUT, geom_only=a.geom_only)
        if ind_name is None:
            skipped.append(f"{L}({sim_path})"); print(f"  L={L:2d}  SKIP ({sim_path})"); continue
        model_dir = str(Path(sim_path).resolve().relative_to(REPO))
        rows.append(dict(idx=len(rows) + 1, target_nH=L, op_GHz=a.freq, N=N,
                         w_um=round(d["w"] * 1e6, 4), s_um=round(d["s"] * 1e6, 4),
                         d_out_um=round(d["d_out"] * 1e6, 3), d_avg_um=round(d["d_avg"] * 1e6, 3),
                         gp_L_nH=round(d["L"] * 1e9, 4), gp_Q=round(d["Q"], 3),
                         gp_SRF_GHz=round(d["f_sr"] / 1e9, 3), model_dir=model_dir))
        _write_manifest(OUT, rows)
        print(f"  [{len(rows):2d}] L={L:2d}nH  N={N} w={d['w']*1e6:5.1f} s={d['s']*1e6:5.1f} "
              f"d_out={d['d_out']*1e6:5.0f}  Q={d['Q']:5.1f}  SRF={d['f_sr']/1e9:4.2f}  -> {ind_name}")

    print(f"\nWrote {len(rows)} models + manifest ({len(skipped)} skipped): {OUT/'manifest.csv'}")
    if skipped:
        print("  skipped:", ", ".join(skipped))


if __name__ == "__main__":
    main()
