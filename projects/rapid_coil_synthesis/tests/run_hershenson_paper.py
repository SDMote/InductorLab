"""
Runs InductanceTargetMaxQ against Hershenson's published Table 1 designs and
prints our solved geometry, Q and SRF next to the published values.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hershenson_paper_pdk import HERSHENSON_PAPER

from rapid_coil_synthesis.gp.hershenson import InductanceTargetMaxQ
import rapid_coil_synthesis.gp.mohan as mohan
import numpy as np

# Hershenson's own fitted square-inductor coefficients (paper eq. 2), which differ
# from mohan.py's SQUARE (Mohan's own paper). Patched in here at runtime only, for
# reproducing her published numbers, without touching the shared mohan.py constants.
mohan.SHAPES["square"] = mohan.ShapeParams(
    sides=4,
    inductance=mohan.InductanceCoefficients(
        beta=1.66e-3, a1=-1.33, a2=-0.125, a3=2.50, a4=1.83, a5=-0.022
    ),
)

TABLE_1 = {
    "L1": dict(n=4.75, w=7.8,  s=1.9, d_out=206.3, Ls=6,  freq_hz=2.5e9, srf_min_hz=7e9),
    "L2": dict(n=7.5,  w=3.2,  s=1.9, d_out=166.5, Ls=12, freq_hz=2.5e9, srf_min_hz=7e9),
    "L3": dict(n=9.5,  w=1.9,  s=1.9, d_out=152.9, Ls=18, freq_hz=2.5e9, srf_min_hz=7e9),
    "L4": dict(n=8,    w=4.3,  s=1.9, d_out=221.6, Ls=18, freq_hz=2.5e9, srf_min_hz=0),
    "L5": dict(n=3.75, w=13,   s=1.9, d_out=292.2, Ls=6,  freq_hz=1.5e9, srf_min_hz=5e9),
    "L6": dict(n=6.5,  w=5.4,  s=1.9, d_out=216.7, Ls=12, freq_hz=1.5e9, srf_min_hz=5e9),
}

row_fmt = "{:4} {:>7} {:>8} {:>7} {:>10} {:>8} {:>7} {:>9}"
print(row_fmt.format("", "n", "w[um]", "s[um]", "d_out[um]", "L_s[nH]", "Q", "SRF[GHz]"))
print(row_fmt.format("", "-"*7, "-"*8, "-"*7, "-"*10, "-"*8, "-"*7, "-"*9))

for name, spec in TABLE_1.items():
    target = InductanceTargetMaxQ(HERSHENSON_PAPER, ind_nh=spec["Ls"], freq_hz=spec["freq_hz"],
                                   srf_min_hz=spec["srf_min_hz"], pgs=True, single_ended=True)
    try:
        target.solve()
    except Exception:
        print(row_fmt.format(name, "infeasible", "", "", "", "", "", ""))
        continue

    sol = target.solution["variables"]
    n_v = sol[target.n]
    w_v = target.w.sub(sol).value.to("um").magnitude
    s_v = target.s.sub(sol).value.to("um").magnitude
    dout_v = target.d_out.sub(sol).value.to("um").magnitude
    Ls_v = target.L_s.sub(sol).value.to("nH").magnitude
    Q_v = sol[target.Q_min]

    Rs_v = target.R_s.sub(sol).value.to("ohm").magnitude
    Cs_v = target.C_s.sub(sol).value.to("F").magnitude
    Cp_v = target.C_p.sub(sol).value.to("F").magnitude
    Ctot_v = Cs_v + Cp_v  # one-port total (single_ended=True): Ctot = Cs + Cp, no factor of 2
    Ls_h = target.L_s.sub(sol).value.to("H").magnitude
    omega_sr = np.sqrt(1/(Ls_h*Ctot_v) - Rs_v**2/Ls_h**2)
    srf_v = omega_sr / (2*np.pi) / 1e9

    print(row_fmt.format(name, f"{n_v:.3f}", f"{w_v:.3f}", f"{s_v:.3f}",
                          f"{dout_v:.2f}", f"{Ls_v:.3f}", f"{Q_v:.3f}", f"{srf_v:.3f}"))

print()
print("Published (Table 1):")
for name, spec in TABLE_1.items():
    print(row_fmt.format(name, f"{spec['n']:.3f}", f"{spec['w']:.3f}", f"{spec['s']:.3f}",
                          f"{spec['d_out']:.2f}", f"{spec['Ls']:.3f}", "", ""))
