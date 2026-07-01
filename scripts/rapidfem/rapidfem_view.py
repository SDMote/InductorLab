"""
rapidfem_view.py -- visualise the rapidfem inductor model in the rapidfem UI.

Run the UI server (from the repo root), then open the printed URL and hit the
"run" button on this file in the browser editor:

    LD_LIBRARY_PATH=$HOME/.pyenv/versions/InductorLab/lib:$LD_LIBRARY_PATH \
    RAPIDFEM_SOLVER=pardiso \
    python3 -m rapidfem serve scripts/

It builds exactly the geometry scripts/simulate_rapidfem.py solves (same
build_model), shows the pre-mesh OCC geometry, meshes, then shows the tet mesh.
Edit GDS / STACK_XML below to inspect a different model.
"""
from pathlib import Path

import rapidfem as rf

import simulate_rapidfem as sim

# --- choose what to view -----------------------------------------------------
GDS = sim.REPO / "scripts/.out/sweep_2p5ghz/sweep_L1_N1_134p26_2p0_711p85_forEM.gds"
STACK_XML = "resources/SG13G2_200um.xml"   # or "builtin"
MAXH, METAL_MAXH, PAD, AIR = 120.0, 50.0, 1.1, 30.0
# -----------------------------------------------------------------------------

stack = sim.make_stack(STACK_XML)
g, port_plates, info = sim.build_model(
    Path(GDS), stack, pad=PAD, air_thick=AIR, maxh=MAXH, metal_maxh=METAL_MAXH)

print("ports (um):", [(round(x * 1e6, 1), round(y * 1e6, 1)) for x, y, _ in info["ports"]])
rf.show(g, "geometry")          # pre-mesh: OCC surfaces (coil, ground, ports, substrate, air)

g.mesh(maxh=MAXH * 1e-6)
rf.show(g, "mesh")              # post-mesh: tet mesh
print("meshed; scrub the viewer to inspect the coil / ports / ground frame")
