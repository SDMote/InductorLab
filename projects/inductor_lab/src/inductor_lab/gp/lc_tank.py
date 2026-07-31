"""
Geometric program for an LC resonator tank inductor.

Direct translation of the "Inductors for LC resonators" section in
notebooks/GeometricProgramming.ipynb.

Objective: minimise 1/R_tank (maximise tank resistance) subject to resonance
at the operating frequency, a minimum tank Q, series resistance bound, and
PDK dimensional constraints.  The additional tank capacitance C_ad (pad,
transistor parasitics, etc.) is passed as a parameter.

The paper (Jaramillo & Maksimovic, CrystalFreeIoT'26) does not give separate LC-tank
equations -- it only states the tool can size LC-tank inductors -- so this reuses the SAME
series/shunt element models as spiral.design_inductor (Sec. 3/4: R_series skin-effect term,
C_ox/C_s/C_si/R_si, and the Sec. 4.2 iterative kappa_R/kappa_C fixed point for R_p/C_p), just
under a different objective (no L target; maximise tank resistance instead of Q).

Uses single-ended topology: C_tot = C_p + C_s.
No inductance target -- the GP is free to choose L to maximise R_tank.
"""

import math
from dataclasses import dataclass

from gpkit import Model, Variable

from inductor_lab.gp.spiral import (
    InductorDesign,
    _ALPHA_1, _ALPHA_2, _ALPHA_3, _ALPHA_4, _ALPHA_5,
    _BETA, _EPS_0, _MU_0,
    _rsi_fit, _series_resistance_k, _shunt_elements, _yue_factors,
)


# -- Result type --------------------------------------------------------------

@dataclass
class LCTankDesign(InductorDesign):
    """LC tank inductor with tank-level performance metrics."""
    L_tank: float   # effective tank inductance [H]
    C_tank: float   # total tank capacitance = C_ad + C_tot [F]
    R_tank: float   # tank parallel resistance [ohm]
    Q_tank: float   # tank quality factor


# -- Solver -------------------------------------------------------------------

def design_lc_tank(
    c_ad_farads: float,
    frequency_hz: float,
    pdk,
    min_q: float = 1.4,
) -> LCTankDesign:
    """Maximise R_tank for an LC resonator inductor via GP.

    Args:
        c_ad_farads:  additional tank capacitance (pads, transistors, ...) [F]
        frequency_hz: resonance frequency [Hz]
        pdk:          PDK parameters dataclass (SG13G2Params or compatible)
        min_q:        minimum tank quality factor (default 1.4)

    Returns:
        LCTankDesign with optimised geometry and tank performance.
    """
    omega_val = 2 * math.pi * frequency_hz

    metal = pdk.top_metals[0]

    k1, k3, k8 = _series_resistance_k(pdk, omega_val)
    rsifit = _rsi_fit(pdk)

    # -- gpkit constants ------------------------------------------------------
    omega_c   = Variable("omega",    omega_val,  "rad/s", constant=True)
    C_ad_c    = Variable("C_ad",     c_ad_farads, "F",    constant=True)
    Q_tmin_c  = Variable("Q_tmin",   min_q,      "-",     constant=True)

    um_c  = Variable("um",  1e-6, "m",   constant=True)
    nH_c  = Variable("nH",  1e-9, "H",   constant=True)
    m_c   = Variable("m",   1,    "m",   constant=True)
    ohm_c = Variable("ohm", 1,    "ohm", constant=True)
    um2_c = Variable("um2", 1e-12, "m^2", constant=True)

    K1_c = Variable("K1", k1, "ohm", constant=True)

    w_min_c = Variable("w_min", metal.w_min, "m", constant=True)
    s_min_c = Variable("s_min", metal.s_min, "m", constant=True)

    # -- Design variables -----------------------------------------------------
    d_out = Variable("d_out", "m",   positive=True)
    d_avg = Variable("d_avg", "m",   positive=True)
    w     = Variable("w",     "m",   positive=True)
    s     = Variable("s",     "m",   positive=True)
    n     = Variable("n",     "-",   positive=True)
    W_bun = Variable("W_bun", "m",   positive=True)
    R_s   = Variable("R_s",   "ohm", positive=True)

    # -- GP expressions -------------------------------------------------------
    l = 8 * d_avg * n / (1 + 2**0.5)

    L = (_BETA
         * (d_out / um_c) ** _ALPHA_1
         * (w     / um_c) ** _ALPHA_2
         * (d_avg / um_c) ** _ALPHA_3
         * n              ** _ALPHA_4
         * (s     / um_c) ** _ALPHA_5
         * nH_c)

    R_m  = K1_c * l / w
    R_v  = k8 * n * (w / m_c)**(-2) * ohm_c

    C_ox, C_s, C_si, R_si = _shunt_elements(pdk, rsifit, d_out, d_avg, w, s, n, l, W_bun,
                                            um_c, um2_c)

    # R_p, C_p via the same iterative kappa_R/kappa_C fixed point as design_inductor
    # (paper Sec. 4.2, eq. 27/28) -- see spiral._yue_factors.
    kappa_c_val, kappa_r_val = 0.45, 2.5
    sol = None
    for _ in range(40):
        kappaC_c = Variable("kappa_C", kappa_c_val, "-", constant=True)
        kappaR_c = Variable("kappa_R", kappa_r_val, "-", constant=True)
        C_p = kappaC_c * C_ox
        R_p = kappaR_c * R_si

        C_tot    = C_p + C_s                              # single-ended
        L_tank   = (1 + (R_s / (L * omega_c))**2) * L
        C_tank   = C_ad_c + C_tot
        inv_Rt   = 1/R_p + R_s / (L * omega_c)**2

        constraints = [
            L_tank * C_tank    <= omega_c**(-2),
            omega_c * L_tank * inv_Rt <= 1 / Q_tmin_c,
            R_s >= R_m + R_v,
            w   >= w_min_c,
            s   >= s_min_c,
            d_avg + n*s + n*w <= d_out,
            n*w/W_bun + n*s/W_bun <= 1,
        ]

        sol = Model(inv_Rt, constraints).solve(verbosity=0)
        sv = sol["variables"]
        w_v  = w.sub(sv).value.to("m").magnitude
        s_v  = s.sub(sv).value.to("m").magnitude
        n_v  = float(n.sub(sv).value)
        do_v = d_out.sub(sv).value.to("m").magnitude
        da_v = d_avg.sub(sv).value.to("m").magnitude
        kappaC_new, kappaR_new = _yue_factors(pdk, omega_val, w_v, s_v, n_v, do_v, da_v)
        converged = (abs(kappaC_new - kappa_c_val) <= 1e-3 * kappa_c_val and
                     abs(kappaR_new - kappa_r_val) <= 1e-3 * kappa_r_val)
        kappa_c_val = 0.5 * kappaC_new + 0.5 * kappa_c_val
        kappa_r_val = 0.5 * kappaR_new + 0.5 * kappa_r_val
        if converged:
            break

    # -- Extract solution ------------------------------------------------------
    def _q(expr):
        return expr.sub(sol["variables"]).value

    n_val      = float(_q(n))
    w_val      = _q(w).to("m").magnitude
    s_val      = _q(s).to("m").magnitude
    d_out_val  = _q(d_out).to("m").magnitude
    d_avg_val  = _q(d_avg).to("m").magnitude
    l_val      = _q(l).to("m").magnitude
    L_val      = _q(L).to("H").magnitude
    Rs_val     = _q(R_s).to("ohm").magnitude
    C_tot_val  = _q(C_tot).to("F").magnitude
    L_tank_val = _q(L_tank).to("H").magnitude
    C_tank_val = _q(C_tank).to("F").magnitude
    inv_Rt_val = _q(inv_Rt).to("S").magnitude
    d_in_val   = d_avg_val - (n_val - 1) * s_val - n_val * w_val

    R_tank_val = 1.0 / inv_Rt_val
    Q_tank_val = R_tank_val / (omega_val * L_tank_val)
    Q_ind_val  = omega_val * L_val / Rs_val

    omega_sr_val = math.sqrt(2 / (L_val * C_tot_val) - Rs_val**2 / L_val**2)
    f_sr_val     = omega_sr_val / (2 * math.pi)

    return LCTankDesign(
        n=n_val, w=w_val, s=s_val,
        d_out=d_out_val, d_avg=d_avg_val, d_in=d_in_val,
        l=l_val, Q=Q_ind_val, L=L_val, R_s=Rs_val,
        R_m=_q(R_m).to("ohm").magnitude,
        R_v=_q(R_v).to("ohm").magnitude,
        R_p=_q(R_p).to("ohm").magnitude,
        R_si=_q(R_si).to("ohm").magnitude,
        C_ox=_q(C_ox).to("F").magnitude,
        C_s=_q(C_s).to("F").magnitude,
        C_si=_q(C_si).to("F").magnitude,
        C_p=_q(C_p).to("F").magnitude,
        C_tot=C_tot_val,
        f_sr=f_sr_val,
        L_tank=L_tank_val,
        C_tank=C_tank_val,
        R_tank=R_tank_val,
        Q_tank=Q_tank_val,
    )
