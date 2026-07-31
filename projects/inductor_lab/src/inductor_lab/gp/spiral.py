"""
Geometric program for octagonal spiral inductors.

Direct translation of Sections 3 and 4 ("Target Geometry and Proposed Model" / "Geometric
Programming Model") of Jaramillo & Maksimovic, "A Tool for On-Chip Inductor Design Using
Geometric Programming", CrystalFreeIoT'26 -- every fitted constant below (delta=0.36 for
inter-turn capacitance, Aup/pup for underpass, Asp/psp for oxide fringing, beta_sub/tau_self for
substrate self-capacitance, beta for the inductance monomial) is the paper's own published value,
not re-derived here. See fringing.py for the (paper-exact) inter-turn/underpass geometry
identities and the R_si legalization.

R_series (paper eq. 13) is implemented as its skin-effect term only; the current-crowding
correction [1 + (f/fcrit)^2/10] (eq. 14) is intentionally OMITTED -- the paper publishes no
fitted legalization constants for it (unlike C_it/C_ox/C_si, which each get an explicit
Table/eq with numeric coefficients), and a from-scratch GP-legal bound for it materially
distorted results (pushed the 10 nH/2.4 GHz design to ~1/3 the paper's published width and
roughly half its Q).

Both topologies share the same GP structure; the only difference is a factor of 2 (cs_factor)
that multiplies C_s, and divides gamma, delta, and R_p:
  Topology.DIFFERENTIAL:  cs_factor = 2  (C_tot = C_p + 2*C_s)
  Topology.SINGLE_ENDED:  cs_factor = 1  (C_tot = C_p + C_s)
"""

import enum
import math
from dataclasses import dataclass

from gpkit import Model, Variable

from .fringing import A_SELF_COEF_PAPER, BETA_SUB_PAPER, c_si_paper, fit_rsi_paper, g_si_paper

_MU_0  = 4 * math.pi * 1e-7   # H/m
_EPS_0 = 8.854187e-12          # F/m

# Modified Wheeler coefficients for octagonal inductors (Mohan et al. [11]). The paper "retains
# the alpha_i coefficients presented in [11]" unchanged and only refits the magnitude
# coefficient beta = 1.222e-3 for the symmetric SG13G2 target geometry (Sec. 4.1).
_BETA    =  1.222e-3
_ALPHA_1 = -1.21
_ALPHA_2 = -0.163
_ALPHA_3 =  2.43
_ALPHA_4 =  1.75
_ALPHA_5 = -0.049

# -- Paper eq. 22/23, Table 1: inter-turn capacitance legalization (delta=0.36 fringe factor) --
_IT_AL, _IT_PW, _IT_PS = 7.2962e-11, 0.1642, -0.78702   # C_l[F/m] = Al*w[um]^pw*s[um]^ps
_IT_AD, _IT_PD         = 0.13344, 1.35062               # weight of d_avg term
_IT_AW, _IT_PW2        = 0.016814, 2.91993              # weight of w term
_IT_AS, _IT_PS2        = 0.073753, 2.32659              # weight of s term

# -- Paper eq. 24: underpass capacitance turn-factor legalization -----------------------------
_UP_A, _UP_P = 0.134, 1.348

# -- Paper eq. 25/26: oxide fringing-perimeter capacitance legalization (Asp at W_bun in um) --
_OX_ASP, _OX_PSP = 1.6842e-11, 0.08904


@dataclass(frozen=True)
class InductanceCoefficients:
    """Monomial coefficients for the Mohan inductance expression

        L_nH = beta * d_out^a1 * w^a2 * d_avg^a3 * n^a4 * s^a5

    with every length in microns and L in nanohenries.  The defaults
    (``MOHAN_OCTAGONAL``) are the paper's own fitted coefficients (Mohan's exponents, beta
    refit to 1.222e-3, Sec. 4.1); pass a process-specific fit -- e.g. one produced by
    asitic_sg13g2_sweep.py -- to override them for a particular PDK and layout style.
    """
    beta: float
    a1:   float   # exponent of d_out
    a2:   float   # exponent of w
    a3:   float   # exponent of d_avg
    a4:   float   # exponent of n
    a5:   float   # exponent of s


#: The paper's own fitted octagonal coefficients (Sec. 4.1, mean error 0.51%, RMS 9.46%).
MOHAN_OCTAGONAL = InductanceCoefficients(
    beta=_BETA, a1=_ALPHA_1, a2=_ALPHA_2, a3=_ALPHA_3, a4=_ALPHA_4, a5=_ALPHA_5,
)


class Topology(enum.Enum):
    DIFFERENTIAL = "differential"
    SINGLE_ENDED = "single_ended"


# -- Shared element-model helpers (also used by lc_tank.py) -------------------

def _series_resistance_k(pdk, omega_val):
    """Metal (eq. 13, skin-effect term only -- see module docstring on the current-crowding
    term), underpass (eq. 9/24) and via-array series-element coefficients."""
    metal = pdk.top_metals[0]
    via   = pdk.top_vias[0]
    sigma_m, t_m = metal.sigma, metal.thickness
    sigma_v, t_via, a_via, b_via = via.sigma, via.thickness, via.width, via.space
    e_ox_val = pdk.eps_r_ox * _EPS_0

    skin_m = math.sqrt(2 / (omega_val * _MU_0 * sigma_m))
    if skin_m < t_m:
        k1 = 1.0 / (sigma_m * skin_m * (1 - math.exp(-t_m / skin_m)))
    else:
        k1 = 1.0 / (sigma_m * t_m)

    k3 = e_ox_val / t_via   # F/m^2  underpass capacitance per area

    skin_v = math.sqrt(2 / (omega_val * _MU_0 * sigma_v))
    if skin_v < a_via:
        k8 = 2 * t_via / (
            sigma_v * a_via * skin_v
            * (1 - math.exp(-a_via / skin_v))
            * (a_via + b_via)**2
        )
    else:
        k8 = 2 * t_via * (a_via + b_via)**2 / (sigma_v * a_via**2)

    return k1, k3, k8


def _rsi_fit(pdk):
    """R_si legalization (paper eq. 20), see fringing.fit_rsi_paper."""
    metal = pdk.top_metals[0]
    return fit_rsi_paper(pdk, n_lo=2.0, n_hi=9.0, w_lo_um=metal.w_min * 1e6, w_hi_um=28.0,
                         s_lo_um=2.0, s_hi_um=18.0)


def _shunt_elements(pdk, rsifit, d_out, d_avg, w, s, n, l, W_bun, um_c, um2_c):
    """Builds the GP expressions for C_ox, C_s, C_si, R_si (paper eq. 6/12/19/20/26)."""
    tan8 = math.tan(math.pi / 8.0)
    e_ox_val  = pdk.eps_r_ox  * _EPS_0
    t_ox_val  = pdk.t_ox
    e_sub_val = pdk.eps_r_sub * _EPS_0
    t_sub_val = pdk.t_sub
    metal = pdk.top_metals[0]
    via   = pdk.top_vias[0]
    k3 = e_ox_val / via.thickness

    # C_ox: paper Sec. 3.4/eq. 26, Scuderi-style solid-disk area + edge-fringe perimeter. C_A
    # (solid-disk area, EXACT octagon identity, no fit) and the shared 8*tan(pi/8) octagon
    # factor already fold in the paper's own /2 from C_ox=(C_A+C_P)/2.
    e_ox_over_tox_c = Variable("e_ox_over_tox", e_ox_val / t_ox_val, "F/m^2", constant=True)
    Asp_c = Variable("Asp", _OX_ASP, "F/m", constant=True)
    C_A_term = 4.0 * tan8 * e_ox_over_tox_c * d_avg * n * (w + s)
    C_P_term = 8.0 * tan8 * d_avg * Asp_c * (W_bun / um_c) ** _OX_PSP
    C_ox = C_A_term + C_P_term

    # C_s = C_underpass + C_interturn, paper eq. 12/24 and eq. 6/22-23 (Table 1).
    K3_c = Variable("K3", k3, "F/m^2", constant=True)
    C_up = K3_c * _UP_A * n**_UP_P * w**2
    Cl_A_c = Variable("Cl_A", _IT_AL, "F/m", constant=True)
    C_l_term = Cl_A_c * (w / um_c) ** _IT_PW * (s / um_c) ** _IT_PS
    C_it = (8.0 * tan8 * C_l_term
            * (_IT_AD * n**_IT_PD * d_avg + _IT_AW * n**_IT_PW2 * w + _IT_AS * n**_IT_PS2 * s))
    C_s = C_up + C_it

    # C_si: paper eq. 19, rim self-cap + under-metal parallel-plate term.
    Beta_sub_c = Variable("Beta_sub", BETA_SUB_PAPER, "F/m", constant=True)
    e_sub_over_tsub_c = Variable("e_sub_over_tsub", e_sub_val / t_sub_val, "F/m^2",
                                 constant=True)
    C_si = Beta_sub_c * (A_SELF_COEF_PAPER * d_out) + e_sub_over_tsub_c * l * w

    # R_si: paper eq. 20, legalized via rsifit.
    Rsi_A_c = Variable("Rsi_A", rsifit.Rsi_A, "ohm", constant=True)
    R_si = Rsi_A_c * (d_out / um_c) ** rsifit.Rsi_pd * (l * w / um2_c) ** rsifit.Rsi_pa

    return C_ox, C_s, C_si, R_si


def _yue_factors(pdk, omega_val, w_v, s_v, n_v, d_out_v, d_avg_v):
    """Paper eq. 27/28: exact Rp/Cp at a frozen geometry, returned as the dimensionless ratios
    kappa_C=Cp/Cox, kappa_R=Rp/Rsi the GP freezes each iteration (Sec. 4.2)."""
    tan8 = math.tan(math.pi / 8.0)
    e_ox_val = pdk.eps_r_ox * _EPS_0
    t_ox_val = pdk.t_ox
    l_v = 8.0 * d_avg_v * n_v / (1.0 + 2.0**0.5)
    Wbun_v = n_v * w_v + n_v * s_v
    C_A_v = 4.0 * tan8 * (e_ox_val / t_ox_val) * d_avg_v * n_v * (w_v + s_v)
    C_P_v = 8.0 * tan8 * d_avg_v * _OX_ASP * (Wbun_v * 1e6) ** _OX_PSP
    C_ox_v = C_A_v + C_P_v
    C_si_v = c_si_paper(pdk, d_out_v, l_v, w_v)
    R_si_v = 1.0 / g_si_paper(pdk, d_out_v, l_v, w_v)
    wR_v   = omega_val * R_si_v
    C_p_v  = (C_ox_v * (1 + wR_v**2 * C_si_v * (C_ox_v + C_si_v))
              / (1 + wR_v**2 * (C_ox_v + C_si_v)**2))
    R_p_v  = (R_si_v * (C_ox_v + C_si_v)**2 / C_ox_v**2
              + 1 / (omega_val**2 * C_ox_v**2 * R_si_v))
    return C_p_v / C_ox_v, R_p_v / R_si_v          # kappa_C, kappa_R


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
    integer_turns: bool = True,
    fixed_n: int = None,
    s_max: float = None,
    w_max: float = None,
    davg_max: float = None,
) -> InductorDesign:
    """Maximise Q for an octagonal spiral inductor via GP (paper Sec. 4.3, eq. 32).

    Args:
        nanohenries:  target inductance [nH]
        frequency_hz: operating frequency [Hz]
        pdk:          PDK parameters dataclass (SG13G2Params or compatible)
        topology:     Topology.DIFFERENTIAL (default) or Topology.SINGLE_ENDED
        min_srf_hz:   minimum self-resonance frequency [Hz]
        max_area_m2:  maximum bounding-box area [m^2]; unconstrained if None (default)
        coeffs:       monomial inductance coefficients; defaults to the paper's own Mohan-
                      exponent / refit-beta octagonal fit. Pass a process-specific
                      InductanceCoefficients (e.g. from the ASITIC sweep) to use those instead.
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
    # Integer-turn handling: locate the continuous optimum, then re-solve frozen at
    # floor(n) and ceil(n) and keep the higher-Q design. Skipped when the caller fixes
    # n explicitly or asks for the raw continuous optimum (integer_turns=False).
    if fixed_n is None and integer_turns:
        cont = design_inductor(
            nanohenries, frequency_hz, pdk, topology=topology, min_srf_hz=min_srf_hz,
            max_area_m2=max_area_m2, coeffs=coeffs, integer_turns=False)
        best = None
        for N in sorted({max(1, math.floor(cont.n)), math.ceil(cont.n)}):
            try:
                d = design_inductor(
                    nanohenries, frequency_hz, pdk, topology=topology,
                    min_srf_hz=min_srf_hz, max_area_m2=max_area_m2, coeffs=coeffs,
                    integer_turns=False, fixed_n=N,
                    s_max=s_max, w_max=w_max, davg_max=davg_max)
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

    metal = pdk.top_metals[0]

    k1, k3, k8 = _series_resistance_k(pdk, omega_val)
    rsifit = _rsi_fit(pdk)

    # -- gpkit constants ------------------------------------------------------
    omega_c    = Variable("omega",    omega_val,          "rad/s", constant=True)
    omega_sr_c = Variable("omega_sr", omega_sr_val,       "rad/s", constant=True)
    L_req_c    = Variable("L_req",    nanohenries * 1e-9, "H",     constant=True)

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
    W_bun = Variable("W_bun", "m",   positive=True)   # bundle width n*w+(n-1)*s, for C_ox's
                                                       # fringe term (paper eq. 25/26)
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

    # R_series: paper eq. 13, skin-effect term only (current-crowding correction omitted --
    # see module docstring).
    R_m  = K1_c * l / w
    R_v  = k8 * n * (w / m_c)**(-2) * ohm_c   # notebook unit-normalisation pattern

    C_ox, C_s, C_si, R_si = _shunt_elements(pdk, rsifit, d_out, d_avg, w, s, n, l, W_bun,
                                            um_c, um2_c)

    k_sr  = omega_sr_c / omega_c

    # -- R_p, C_p via the paper's iterative kappa_R/kappa_C fixed point (Sec. 4.2, eq. 27/28) --
    # C_ox is now a posynomial (fringing term) and R_si/C_si are not simple per-area monomials,
    # so R_p/C_p can't enter the GP directly. Following the paper: freeze the dimensionless
    # ratios kappa_C=C_p/C_ox and kappa_R=R_p/R_si, solve the GP, recompute the EXACT Rp/Cp from
    # eq. 27/28 at the new geometry, refresh the ratios, and iterate to a fixed point.
    kappa_c_val, kappa_r_val = 0.45, 2.5      # dimensionless seeds; the fixed point refines
    sol = None
    for _ in range(40):
        kappaC_c = Variable("kappa_C", kappa_c_val, "-", constant=True)
        kappaR_c = Variable("kappa_R", kappa_r_val, "-", constant=True)
        C_p   = kappaC_c * C_ox
        R_p   = kappaR_c * R_si
        rho   = omega_c * L / R_s
        C_tot = C_p + cs_factor * C_s
        gamma = omega_c**2 * L * C_tot / cs_factor
        delta = R_s**2 * C_tot / cs_factor / L
        constraints = [
            L == L_req_c,
            R_s >= R_m + R_v,
            Q_min * (cs_factor*R_p + (rho**2 + 1)*R_s) / (rho * cs_factor*R_p)
                + delta + gamma <= 1,
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
            # Bundle width for C_ox's fringe term (paper eq. 25/26). Exact is n*w+(n-1)*s, but
            # the -s is signomial; using n*(w+s) over-estimates W_bun by one gap, which -- since
            # C_ox rises with W_bun (psp>0) and nothing rewards a larger C_ox -- the solver
            # drives to this lower bound, so it stays tight (same pattern as the d_avg bound).
            n*w/W_bun + n*s/W_bun <= 1,
        ]
        if max_area_m2 is not None:
            A_max_c = Variable("A_max", max_area_m2, "m^2", constant=True)
            constraints.append(d_out**2 <= A_max_c)
        # Optional UPPER bounds to keep the design inside a fit's sampled region (the
        # monomial extrapolates otherwise). Monomial <= constant constraints, GP-legal.
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

        sol = Model(Q_min**-1, constraints).solve(verbosity=0)
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
