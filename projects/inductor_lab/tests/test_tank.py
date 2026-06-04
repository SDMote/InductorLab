"""
Smoke test: design an LC tank inductor at 2.4 GHz on IHP SG13G2
with C_ad = 68.55 fF additional capacitance.
Run with:  python tests/test_tank.py
"""

from dataclasses import replace

from inductor_lab.gp.lc_tank import design_lc_tank
from inductor_lab.pdk.sg13g2 import BACKLAPPING, SG13G2

pdk = replace(SG13G2, t_sub=BACKLAPPING[200])

d = design_lc_tank(c_ad_farads=68.55e-15, frequency_hz=2.4e9, pdk=pdk)

print(f"""
Geometry
--------
n      = {d.n:.4f} turns
w      = {d.w * 1e6:.4f} um
s      = {d.s * 1e6:.4f} um
l      = {d.l * 1e6:.4f} um
d_out  = {d.d_out * 1e6:.4f} um
d_avg  = {d.d_avg * 1e6:.4f} um
d_in   = {d.d_in * 1e6:.4f} um

Inductor performance
--------------------
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

Tank performance
----------------
L_tank = {d.L_tank * 1e9:.4f} nH
C_tank = {d.C_tank * 1e15:.4f} fF
R_tank = {d.R_tank / 1e3:.4f} kohm
Q_tank = {d.Q_tank:.4f}
""")
