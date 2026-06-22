"""
Smoke test: design a 10 nH spiral inductor at 2.4 GHz on IHP SG13G2,
for both the differential and single-ended topologies.

design_inductor defaults to integer_turns=True, so the result has a realisable
INTEGER turn count and still predicts the target L exactly (see
integer_turn_resolve.md).
Run with:  python tests/test_spiral.py
"""

from dataclasses import replace

from inductor_lab.gp.spiral import Topology, design_inductor
from inductor_lab.pdk.sg13g2 import BACKLAPPING, SG13G2

pdk = replace(SG13G2, t_sub=BACKLAPPING[200])


def print_design(label, d):
    rho = (d.d_out - d.d_in) / (d.d_out + d.d_in)
    print(f"""
{label}
{"=" * len(label)}
Geometry
--------
n      = {d.n:.4f} turns
w      = {d.w * 1e6:.4f} um
s      = {d.s * 1e6:.4f} um
l      = {d.l * 1e6:.4f} um
d_out  = {d.d_out * 1e6:.4f} um
d_avg  = {d.d_avg * 1e6:.4f} um
d_in   = {d.d_in * 1e6:.4f} um
rho    = {rho:.4f}

Performance
-----------
L      = {d.L * 1e9:.4f} nH
Q      = {d.Q:.4f}
f_sr   = {d.f_sr / 1e9:.4f} GHz

Equivalent circuit
------------------
R_m    = {d.R_m:.4f} ohm
R_v    = {d.R_v:.4f} ohm
R_s    = {d.R_s:.4f} ohm
R_p    = {d.R_p / 1e3:.4f} kohm
R_si   = {d.R_si / 1e3:.4f} kohm
C_ox   = {d.C_ox * 1e15:.4f} fF
C_s    = {d.C_s * 1e15:.4f} fF
C_si   = {d.C_si * 1e15:.4f} fF
C_p    = {d.C_p * 1e15:.4f} fF
C_tot  = {d.C_tot * 1e15:.4f} fF
""")


diff = design_inductor(10, 2.4e9, pdk, topology=Topology.DIFFERENTIAL)
se   = design_inductor(10, 2.4e9, pdk, topology=Topology.SINGLE_ENDED)

print_design("Differential (two-port)", diff)
print_design("Single-ended (one-port)", se)
