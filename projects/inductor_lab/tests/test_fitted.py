"""
Smoke test: design a 10 nH spiral inductor at 2.4 GHz on IHP SG13G2, using the
SG13G2-specific monomial coefficients fitted from the ASITIC symmetric-octagon
(sympoly) sweep instead of Mohan's generic octagonal coefficients.

Mirrors tests/test_spiral.py, but passes a custom InductanceCoefficients so the
GP optimises against the process-specific inductance model.

Run with:  python tests/test_fitted.py
"""

from dataclasses import replace

from inductor_lab.gp.spiral import (
    InductanceCoefficients,
    Topology,
    design_inductor,
)
from inductor_lab.pdk.sg13g2 import BACKLAPPING, SG13G2

pdk = replace(SG13G2, t_sub=BACKLAPPING[200])

# SG13G2 symmetric-octagon (sympoly) AC fit at 2.4 GHz from asitic_sym_sweep.py.
# Source: ASITIC/coefficients_sym_sg13g2_2.4ghz.py  (RMSE 5.37%, N=12581)
# (built-geometry fit; d_avg = d_out - n*w - (n-1)*s, the exact octagon convention)
SG13G2_SYMPOLY = InductanceCoefficients(
    beta=8.613900e-04,
    a1=-1.217735,   # d_out
    a2=-0.141603,   # w
    a3=2.482231,    # d_avg
    a4=1.864156,    # n
    a5=-0.038759,   # s
)


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


diff = design_inductor(10, 2.4e9, pdk, topology=Topology.DIFFERENTIAL,
                       coeffs=SG13G2_SYMPOLY)
se   = design_inductor(10, 2.4e9, pdk, topology=Topology.SINGLE_ENDED,
                       coeffs=SG13G2_SYMPOLY)

print_design("Differential (two-port) -- SG13G2 sympoly fit", diff)
print_design("Single-ended (one-port) -- SG13G2 sympoly fit", se)
