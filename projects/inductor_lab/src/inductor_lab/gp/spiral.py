"""
Geometric program for octagonal spiral inductors.

Direct translation of the "Geometric Programming" and "One port inductors"
sections in notebooks/GeometricProgramming.ipynb.

Both topologies share the same GP structure; the only difference is a factor
of 2 (cs_factor) that multiplies C_s, and divides gamma, delta, and R_p:
  Topology.DIFFERENTIAL:  cs_factor = 2  (C_tot = C_p + 2*C_s)
  Topology.SINGLE_ENDED:  cs_factor = 1  (C_tot = C_p + C_s)
"""

import enum
import math
from dataclasses import dataclass

from gpkit import Model, Variable

_MU_0  = 4 * math.pi * 1e-7   # H/m
_EPS_0 = 8.854187e-12          # F/m

# Modified Wheeler coefficients for octagonal inductors (Mohan et al.)
_BETA    =  1.33e-3
_ALPHA_1 = -1.21
_ALPHA_2 = -0.163
_ALPHA_3 =  2.43
_ALPHA_4 =  1.75
_ALPHA_5 = -0.049


class Topology(enum.Enum):
    DIFFERENTIAL = "differential"
    SINGLE_ENDED = "single_ended"


# -- k-coefficient functions --------------------------------------------------

def _k6(k2: float, k4: float, k5: float, omega: float) -> float:
    """Shunt resistance coefficient [ohm*m^2]. Derived in notebook section Shunt resistance R_p."""
    return k5 * (k2 + k4)**2 / k2**2 + 1.0 / (k2**2 * k5 * omega**2)


def _k7(k2: float, k4: float, k5: float, omega: float) -> float:
    """Shunt capacitance coefficient [F/m^2]. Derived in notebook section Shunt capacitance C_p."""
    return (k2 * (k4 * k5**2 * omega**2 * (k2 + k4) + 1.0)
            / (k5**2 * omega**2 * (k2 + k4)**2 + 1.0))


# -- Result type --------------------------------------------------------------

@dataclass
class InductorDesign:
    """Optimised inductor geometry and equivalent-circuit performance."""
    # Geometry
    n:     float   # number of turns
    w:     float   # track width [m]
    s:     float   # track spacing [m]
    d_out: float   # outer diameter [m]
    d_avg: float   # average diameter [m]
    d_in:  float   # inner diameter [m]
    l:     float   # total conductor length [m]
    # Performance
    Q:     float   # quality factor (GP lower bound)
    L:     float   # inductance [H]
    R_s:   float   # total series resistance [ohm]
    R_m:   float   # metal series resistance [ohm]
    R_v:   float   # via series resistance [ohm]
    R_p:   float   # shunt resistance [ohm]
    R_si:  float   # substrate resistance [ohm]
    C_ox:  float   # oxide capacitance [F]
    C_s:   float   # interwinding capacitance [F]
    C_si:  float   # substrate capacitance [F]
    C_p:   float   # shunt capacitance [F]
    C_tot: float   # total capacitance = C_p + cs_factor*C_s [F]
    f_sr:  float   # self-resonance frequency [Hz]


# -- Solver -------------------------------------------------------------------

def design_inductor(
    nanohenries: float,
    frequency_hz: float,
    pdk,
    topology: Topology = Topology.DIFFERENTIAL,
    min_srf_hz: float = 7e9,
    max_area_m2: float = None,
) -> InductorDesign:
    """Maximise Q for an octagonal spiral inductor via GP.

    Args:
        nanohenries:  target inductance [nH]
        frequency_hz: operating frequency [Hz]
        pdk:          PDK parameters dataclass (SG13G2Params or compatible)
        topology:     Topology.DIFFERENTIAL (default) or Topology.SINGLE_ENDED
        min_srf_hz:   minimum self-resonance frequency [Hz]
        max_area_m2:  maximum bounding-box area [m^2]; unconstrained if None (default)

    Returns:
        InductorDesign with optimised geometry and estimated performance.
    """
    cs_factor = 2 if topology is Topology.DIFFERENTIAL else 1
    omega_val    = 2 * math.pi * frequency_hz
    omega_sr_val = 2 * math.pi * min_srf_hz

    # -- PDK extraction -------------------------------------------------------
    metal = pdk.top_metals[0]
    via   = pdk.top_vias[0]

    sigma_m = metal.sigma
    t_m     = metal.thickness

    sigma_v = via.sigma
    t_via   = via.thickness    # oxide thickness between the two metals (= via height)
    a_via   = via.width
    b_via   = via.space

    e_ox_val  = pdk.eps_r_ox  * _EPS_0
    t_ox_val  = pdk.t_ox
    e_sub_val = pdk.eps_r_sub * _EPS_0
    t_sub_val = pdk.t_sub
    sigma_sub = pdk.sigma_sub

    # -- k-coefficient computation --------------------------------------------
    # k1: series resistance coefficient [ohm], accounts for skin effect
    skin_m = math.sqrt(2 / (omega_val * _MU_0 * sigma_m))
    if skin_m < t_m:
        k1 = 1.0 / (sigma_m * skin_m * (1 - math.exp(-t_m / skin_m)))
    else:
        k1 = 1.0 / (sigma_m * t_m)

    k2 = e_ox_val  / (2 * t_ox_val)    # F/m^2  oxide capacitance per area
    k3 = e_ox_val  / t_via             # F/m^2  interwinding capacitance per area
    k4 = e_sub_val / (2 * t_sub_val)   # F/m^2  substrate capacitance per area
    k5 = 2 * t_sub_val / sigma_sub     # ohm*m^2  substrate resistance x area

    k6 = _k6(k2, k4, k5, omega_val)   # ohm*m^2
    k7 = _k7(k2, k4, k5, omega_val)   # F/m^2

    # k8: via array resistance x area [ohm*m^2]
    skin_v = math.sqrt(2 / (omega_val * _MU_0 * sigma_v))
    if skin_v < a_via:
        k8 = 2 * t_via / (
            sigma_v * a_via * skin_v
            * (1 - math.exp(-a_via / skin_v))
            * (a_via + b_via)**2
        )
    else:
        k8 = 2 * t_via * (a_via + b_via)**2 / (sigma_v * a_via**2)

    # -- gpkit constants ------------------------------------------------------
    omega_c    = Variable("omega",    omega_val,          "rad/s", constant=True)
    omega_sr_c = Variable("omega_sr", omega_sr_val,       "rad/s", constant=True)
    L_req_c    = Variable("L_req",    nanohenries * 1e-9, "H",     constant=True)

    um_c  = Variable("um",  1e-6, "m",   constant=True)
    nH_c  = Variable("nH",  1e-9, "H",   constant=True)
    m_c   = Variable("m",   1,    "m",   constant=True)
    ohm_c = Variable("ohm", 1,    "ohm", constant=True)

    K1_c = Variable("K1", k1, "ohm",     constant=True)
    K2_c = Variable("K2", k2, "F/m^2",   constant=True)
    K3_c = Variable("K3", k3, "F/m^2",   constant=True)
    K4_c = Variable("K4", k4, "F/m^2",   constant=True)
    K5_c = Variable("K5", k5, "ohm*m^2", constant=True)
    K6_c = Variable("K6", k6, "ohm*m^2", constant=True)
    K7_c = Variable("K7", k7, "F/m^2",   constant=True)

    w_min_c = Variable("w_min", metal.w_min, "m", constant=True)
    s_min_c = Variable("s_min", metal.s_min, "m", constant=True)

    # -- Design variables -----------------------------------------------------
    d_out = Variable("d_out", "m",   positive=True)
    d_avg = Variable("d_avg", "m",   positive=True)
    w     = Variable("w",     "m",   positive=True)
    s     = Variable("s",     "m",   positive=True)
    n     = Variable("n",     "-",   positive=True)
    Q_min = Variable("Q_min", "-",   positive=True)
    R_s   = Variable("R_s",   "ohm", positive=True)

    # -- Intermediate GP expressions ------------------------------------------
    l = 8 * d_avg * n / (1 + 2**0.5)   # conductor length (octagon)

    L = (_BETA
         * (d_out / um_c) ** _ALPHA_1
         * (w     / um_c) ** _ALPHA_2
         * (d_avg / um_c) ** _ALPHA_3
         * n              ** _ALPHA_4
         * (s     / um_c) ** _ALPHA_5
         * nH_c)

    R_m  = K1_c * l / w
    C_ox = K2_c * l * w
    C_s  = K3_c * n * w**2
    C_si = K4_c * l * w
    R_si = K5_c / (l * w)
    R_v  = k8 * n * (w / m_c)**(-2) * ohm_c   # notebook unit-normalisation pattern

    R_p  = K6_c / (l * w)
    C_p  = K7_c * l * w

    C_tot = C_p + cs_factor * C_s
    rho   = omega_c * L / R_s
    gamma = omega_c**2 * L * C_tot / cs_factor
    delta = R_s**2 * C_tot / cs_factor / L
    k_sr  = omega_sr_c / omega_c

    # -- Build and solve ------------------------------------------------------
    constraints = [
        L == L_req_c,
        R_s >= R_m + R_v,
        Q_min * (cs_factor*R_p + (rho**2 + 1)*R_s) / (rho * cs_factor*R_p) + delta + gamma <= 1,
        delta / 2 + gamma / 2 <= 1,
        k_sr**2 * gamma / 2 + delta / 2 <= 1,
        w >= w_min_c,
        s >= s_min_c,
        d_avg + n*s + n*w <= d_out,
    ]
    if max_area_m2 is not None:
        A_max_c = Variable("A_max", max_area_m2, "m^2", constant=True)
        constraints.append(d_out**2 <= A_max_c)

    sol = Model(Q_min**-1, constraints).solve()

    # -- Extract solution ------------------------------------------------------
    def _q(expr):
        return expr.sub(sol["variables"]).value

    n_val     = float(_q(n))
    w_val     = _q(w).to("m").magnitude
    s_val     = _q(s).to("m").magnitude
    d_out_val = _q(d_out).to("m").magnitude
    d_avg_val = _q(d_avg).to("m").magnitude
    l_val     = _q(l).to("m").magnitude
    Q_val     = float(_q(Q_min))
    L_val     = _q(L).to("H").magnitude
    Rs_val    = _q(R_s).to("ohm").magnitude
    C_tot_val = _q(C_tot).to("F").magnitude
    d_in_val  = d_avg_val - (n_val - 1) * s_val - n_val * w_val

    omega_sr_val = math.sqrt(2 / (L_val * C_tot_val) - Rs_val**2 / L_val**2)
    f_sr_val     = omega_sr_val / (2 * math.pi)

    return InductorDesign(
        n=n_val, w=w_val, s=s_val,
        d_out=d_out_val, d_avg=d_avg_val, d_in=d_in_val,
        l=l_val, Q=Q_val, L=L_val, R_s=Rs_val,
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
    )
