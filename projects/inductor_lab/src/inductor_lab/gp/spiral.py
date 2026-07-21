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

from .fringing import ShuntCapCoefficients, fit_shunt_cap

_MU_0  = 4 * math.pi * 1e-7   # H/m
_EPS_0 = 8.854187e-12          # F/m

# Modified Wheeler coefficients for octagonal inductors (Mohan et al.)
_BETA    =  1.33e-3
_ALPHA_1 = -1.21
_ALPHA_2 = -0.163
_ALPHA_3 =  2.43
_ALPHA_4 =  1.75
_ALPHA_5 = -0.049


@dataclass(frozen=True)
class InductanceCoefficients:
    """Monomial coefficients for the Mohan inductance expression

        L_nH = beta * d_out^a1 * w^a2 * d_avg^a3 * n^a4 * s^a5

    with every length in microns and L in nanohenries.  The defaults
    (``MOHAN_OCTAGONAL``) are Mohan et al.'s generic octagonal fit; pass a
    process-specific fit -- e.g. one produced by asitic_sg13g2_sweep.py -- to
    override them for a particular PDK and layout style.
    """
    beta: float
    a1:   float   # exponent of d_out
    a2:   float   # exponent of w
    a3:   float   # exponent of d_avg
    a4:   float   # exponent of n
    a5:   float   # exponent of s


#: Mohan et al. generic octagonal coefficients (the historical default).
MOHAN_OCTAGONAL = InductanceCoefficients(
    beta=_BETA, a1=_ALPHA_1, a2=_ALPHA_2, a3=_ALPHA_3, a4=_ALPHA_4, a5=_ALPHA_5,
)


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
    coeffs: InductanceCoefficients = MOHAN_OCTAGONAL,
    shunt_cap: ShuntCapCoefficients = None,
    integer_turns: bool = True,
    fixed_n: int = None,
    s_max: float = None,
    w_max: float = None,
    davg_max: float = None,
) -> InductorDesign:
    """Maximise Q for an octagonal spiral inductor via GP.

    Args:
        nanohenries:  target inductance [nH]
        frequency_hz: operating frequency [Hz]
        pdk:          PDK parameters dataclass (SG13G2Params or compatible)
        topology:     Topology.DIFFERENTIAL (default) or Topology.SINGLE_ENDED
        min_srf_hz:   minimum self-resonance frequency [Hz]
        max_area_m2:  maximum bounding-box area [m^2]; unconstrained if None (default)
        coeffs:       monomial inductance coefficients; defaults to the Mohan
                      octagonal fit. Pass a process-specific InductanceCoefficients
                      (e.g. from the ASITIC sweep) to use those instead.
        integer_turns: if True (default) return a realisable INTEGER-turn design -- the
                      GP is solved continuously to locate n, then re-solved frozen at
                      floor(n) and ceil(n), keeping the higher-Q result, so the design
                      has integer turns AND hits the target L exactly (see
                      PROJECT_OVERVIEW.md (sec. 2)). Set False for the raw continuous
                      optimum. Ignored when ``fixed_n`` is given.
        fixed_n:      if given, freeze the turn count at this integer (a monomial
                      equality ``n == fixed_n``) and let the solver re-optimise
                      w/s/d_out so the design hits the target L with that many turns.

    Returns:
        InductorDesign with optimised geometry and estimated performance.
    """
    # Bundled Hasegawa fringing shunt-cap fit (see fringing.py). Fit once from the
    # PDK here and forward it through the recursion so it isn't re-fit per solve.
    if shunt_cap is None:
        shunt_cap = fit_shunt_cap(pdk)

    # Integer-turn handling: locate the continuous optimum, then re-solve frozen at
    # floor(n) and ceil(n) and keep the higher-Q design. Skipped when the caller fixes
    # n explicitly or asks for the raw continuous optimum (integer_turns=False).
    if fixed_n is None and integer_turns:
        cont = design_inductor(
            nanohenries, frequency_hz, pdk, topology=topology, min_srf_hz=min_srf_hz,
            max_area_m2=max_area_m2, coeffs=coeffs, shunt_cap=shunt_cap,
            integer_turns=False)
        best = None
        for N in sorted({max(1, math.floor(cont.n)), math.ceil(cont.n)}):
            try:
                d = design_inductor(
                    nanohenries, frequency_hz, pdk, topology=topology,
                    min_srf_hz=min_srf_hz, max_area_m2=max_area_m2, coeffs=coeffs,
                    shunt_cap=shunt_cap, integer_turns=False, fixed_n=N)
            except Exception:
                continue
            if best is None or d.Q > best.Q:
                best = d
        if best is None:
            raise RuntimeError(f"no feasible integer-turn design for {nanohenries} nH")
        return best

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
    # k7 (parallel-plate shunt cap coeff) retired: C_p now comes from the bundled
    # Hasegawa fringing fit (see C_p below and fringing.py). _k7 kept for reference.

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

    w_min_c = Variable("w_min", metal.w_min, "m", constant=True)
    s_min_c = Variable("s_min", metal.s_min, "m", constant=True)

    # Bundled fringing shunt-cap fit:  C'_ser[fF/um] = Acap * (W_bun[um])^bcap
    Acap_c = Variable("Acap", shunt_cap.A, "-", constant=True)
    fF_c   = Variable("fF",   1e-15,       "F", constant=True)

    # -- Design variables -----------------------------------------------------
    d_out = Variable("d_out", "m",   positive=True)
    d_avg = Variable("d_avg", "m",   positive=True)
    w     = Variable("w",     "m",   positive=True)
    s     = Variable("s",     "m",   positive=True)
    n     = Variable("n",     "-",   positive=True)
    W_bun = Variable("W_bun", "m",   positive=True)   # bundle width n*w+(n-1)*s
    Q_min = Variable("Q_min", "-",   positive=True)
    R_s   = Variable("R_s",   "ohm", positive=True)

    # -- Intermediate GP expressions ------------------------------------------
    l = 8 * d_avg * n / (1 + 2**0.5)   # conductor length (octagon)

    L = (coeffs.beta
         * (d_out / um_c) ** coeffs.a1
         * (w     / um_c) ** coeffs.a2
         * (d_avg / um_c) ** coeffs.a3
         * n              ** coeffs.a4
         * (s     / um_c) ** coeffs.a5
         * nH_c)

    R_m  = K1_c * l / w
    C_ox = K2_c * l * w
    C_s  = K3_c * n * w**2
    C_si = K4_c * l * w
    R_si = K5_c / (l * w)
    R_v  = k8 * n * (w / m_c)**(-2) * ohm_c   # notebook unit-normalisation pattern

    R_p  = K6_c / (l * w)

    # Shunt capacitance from the bundled Hasegawa fringing model (fringing.py):
    # C_p = C'_ser(W_bun) * l_bundle, with C'_ser[fF/um] = Acap*(W_bun[um])^bcap
    # and l_bundle = l/n = 3.3137*d_avg (ONE loop perimeter, not the n-turn wire).
    # Replaces the old parallel-plate C_p = K7_c*l*w, which ignored fringing and
    # under-predicted the substrate coupling several-fold (see PROJECT_OVERVIEW
    # sec 7.3 / memory srf-root-cause-cp).
    l_bundle = l / n
    C_p = Acap_c * (W_bun / um_c)**shunt_cap.b * (l_bundle / um_c) * fF_c

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
        # Average-diameter / fit bound. The exact octagon build-up is
        # n*w + (n-1)*s (n traces, n-1 gaps per side), so d_avg = d_out - n*w
        # - (n-1)*s -- the SAME convention the ASITIC coefficients are fit on
        # (built geometry). But (n-1)*s is a signomial and is rejected by a pure
        # GP, so we keep the canonical Boyd/Hershenson n*(w+s) form here: it is
        # tighter by one s (~0.7% of d_avg), i.e. conservative. The exact d_avg
        # is recovered below in the post-solve d_in back-out.
        d_avg + n*s + n*w <= d_out,
        # Bundle width for the fringing shunt cap. Exact is n*w + (n-1)*s, but the
        # -s is signomial; using n*(w+s) over-estimates W_bun by one gap. Since
        # C_p rises with W_bun (bcap > 0) and nothing rewards a larger C_p, the
        # solver drives W_bun to this lower bound -> tight, and the +s over-count
        # is conservative (slightly higher C_p / lower SRF). Posynomial <= 1.
        n*w/W_bun + n*s/W_bun <= 1,
    ]
    if max_area_m2 is not None:
        A_max_c = Variable("A_max", max_area_m2, "m^2", constant=True)
        constraints.append(d_out**2 <= A_max_c)
    # Optional UPPER bounds to keep the design inside a fit's sampled region (the
    # monomial extrapolates otherwise -- e.g. the rapidfem LHS only sampled
    # s in [2,7] um, but Q rises with s so the optimizer walks s past that).
    # Monomial <= constant constraints, GP-legal.
    if s_max is not None:
        constraints.append(s <= Variable("s_max", s_max, "m", constant=True))
    if w_max is not None:
        constraints.append(w <= Variable("w_max", w_max, "m", constant=True))
    if davg_max is not None:
        constraints.append(d_avg <= Variable("davg_max", davg_max, "m", constant=True))
    if fixed_n is not None:
        # Freeze the turn count at an integer; the solver re-sizes w/s/d_out so the
        # inductance monomial still hits L_req. Monomial equality -> stays a GP.
        constraints.append(n == fixed_n)

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
