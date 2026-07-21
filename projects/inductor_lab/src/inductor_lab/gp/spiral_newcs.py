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

from .fringing import (ShuntCapCoefficients, fit_shunt_cap, fit_interturn_paper,
                        fit_interturn_derived, fit_cox_derived, c_ox_derived,
                        fit_cox_scuderi, fit_rsi_derived, BETA_CSI, TAU_SELF_COIL)

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
    s_min: float = None,
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
        s_min:        override the PDK's DRC spacing floor with a HIGHER minimum, e.g. to
                      keep the design inside c_l_interturn_derived's validated range
                      (s>=5um; the physical fringe/sidewall decomposition under-predicts
                      C_it by up to ~2x at tighter spacing -- see FringingMethodology.md).
                      Ignored (PDK floor used) if None or below the PDK's own s_min.

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
            integer_turns=False, s_min=s_min, s_max=s_max, w_max=w_max, davg_max=davg_max)
        best = None
        for N in sorted({max(1, math.floor(cont.n)), math.ceil(cont.n)}):
            try:
                d = design_inductor(
                    nanohenries, frequency_hz, pdk, topology=topology,
                    min_srf_hz=min_srf_hz, max_area_m2=max_area_m2, coeffs=coeffs,
                    shunt_cap=shunt_cap, integer_turns=False, fixed_n=N,
                    s_min=s_min, s_max=s_max, w_max=w_max, davg_max=davg_max)
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
    K3_c = Variable("K3", k3, "F/m^2",   constant=True)
    K4_c = Variable("K4", k4, "F/m^2",   constant=True)
    K5_c = Variable("K5", k5, "ohm*m^2", constant=True)
    K6_c = Variable("K6", k6, "ohm*m^2", constant=True)

    w_min_c = Variable("w_min", metal.w_min, "m", constant=True)
    s_min_val = max(metal.s_min, s_min) if s_min is not None else metal.s_min
    s_min_c = Variable("s_min", s_min_val, "m", constant=True)

    # Bundled fringing shunt-cap fit:  C'_ser[fF/um] = Acap * (W_bun[um])^bcap
    Acap_c = Variable("Acap", shunt_cap.A, "-", constant=True)
    fF_c   = Variable("fF",   1e-15,       "F", constant=True)

    # Physical shunt-branch element models. C_ox: Scuderi-style solid-disk model (paper's
    # "Oxide Capacitance"); C_si/R_si: plate-anchored Green's-function closure (paper's
    # "Substrate Capacitance"/"Substrate Resistance"). These populate the resolved pi branch
    # C_ox -- (R_si || C_si); the GP still drives SRF/Q from the lumped C_p/R_p. See the C_ox/
    # C_si/R_si construction below for the closed forms and the two EM-anchored constants
    # (BETA_CSI, TAU_SELF_COIL; tau_si itself is the pure eps_si/sigma_sub stackup property).

    # Inter-turn series-cap: SAME exact linear-voltage-profile bracket (energy-sum derivation,
    # d_out eliminated via d_avg+n*(w+s)=d_out -> exact 3-term N-weighted sum), but C_l(w,s) is
    # now the FIRST-PRINCIPLES coupled-microstrip derivation (fringing.c_l_interturn_derived:
    # exact sidewall eps_ox*t_m/s + conformal-mapping odd-mode fringe (Cga+Cgd-Cf')/2, no fitted
    # shape) instead of the paper's fitted delta_coeff*w/(w+s). ONE residual constant remains:
    # SCALE_INTERTURN=0.68 (fringing.py), the median EM/derived ratio on 3 reliable coil_cox
    # coils (0.66/0.68/0.68, 3% spread) -- an overall SCALE on an already-derived shape, not a
    # fitted functional form. Physical origin of the 0.68 (an isolated 2-conductor cross-section
    # vs. a real coil's multi-neighbor, curved-perimeter gap) is identified but not yet closed
    # analytically; a floating 3rd-conductor EM test found NO screening effect (falsified),
    # ruling out the simplest version of that mechanism -- see session notes / COX_ONCOIL_HANDOFF.
    # NOTE: validation is thinner here (3 coils) than the paper's own delta=0.36 (broad Palace
    # sweep, median 3%/RMS 14%) -- use fit_interturn_paper for the more broadly-validated model
    # if this one regresses on new data.
    inter = fit_interturn_derived(pdk, n_lo=3.0, n_hi=9.0, w_lo_um=metal.w_min * 1e6, w_hi_um=28.0,
                                  s_lo_um=2.0, s_hi_um=18.0)

    # C_ox: Scuderi-style solid-disk model (paper's "Oxide Capacitance"): per-port
    # C_ox = (C_A+C_P)/2 = 4tan(pi/8)*d_avg*[eps_ox/t_ox*n*(w+s) + 2*Cf(w,s,n)], C_A (solid-disk
    # area) EXACT, only the C_P edge-fringe term (via Cf) needs a 3D log-log monomial fit
    # (fringing.fit_cox_scuderi). Replaces the coupled-microstrip c_ox_derived model: with C_ox
    # now FIXED (not a free 3rd fit parameter) the C_si/R_si degeneracy that previously forced a
    # calibrated tau_eff=62.3ps breaks (scripts/palace/close_csi_rsi.py). Direct-eval RMS
    # 4.8%/10.5% on lhs_backside/sweep_newcs (98/9 coils; C_ox pinned via the 2-param Csi/Rsi
    # refit, not a free fit).
    coxfit = fit_cox_scuderi(pdk, n_lo=2.0, n_hi=9.0, w_lo_um=metal.w_min * 1e6, w_hi_um=28.0,
                             s_lo_um=2.0, s_hi_um=18.0)
    # R_si: plate-anchored Green's-function per-term duality, legalized as a 2D monomial in
    # (d_out, A_metal=l*w) (fringing.fit_rsi_derived) since G_si=1/R_si is a genuine 2-term
    # posynomial (rim self-cap/tau_self + under-metal-plate/tau_si), not itself GP-legal.
    rsifit = fit_rsi_derived(pdk, n_lo=2.0, n_hi=9.0, w_lo_um=metal.w_min * 1e6, w_hi_um=24.0,
                             s_lo_um=2.0, s_hi_um=18.0, davg_lo_um=80.0, davg_hi_um=300.0)

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

    L = (coeffs.beta
         * (d_out / um_c) ** coeffs.a1
         * (w     / um_c) ** coeffs.a2
         * (d_avg / um_c) ** coeffs.a3
         * n              ** coeffs.a4
         * (s     / um_c) ** coeffs.a5
         * nH_c)

    R_m  = K1_c * l / w
    # C_ox: Scuderi-style solid-disk model, see coxfit note above. C_A (solid-disk area) is
    # EXACT (2-term monomial sum, tan8 shared with C_it's own octagon factor); C_P (edge
    # fringe) uses the fitted Cf(n,w,s) monomial.
    tan8 = math.tan(math.pi / 8.0)
    e_ox_c = Variable("e_ox_over_tox", e_ox_val / t_ox_val, "F/m^2", constant=True)
    Cf_A_c = Variable("Cf_A", coxfit.Cf_A, "F/m", constant=True)
    C_A_term = 4.0 * tan8 * e_ox_c * d_avg * n * (w + s)
    C_P_term = (8.0 * tan8 * Cf_A_c * n ** coxfit.Cf_pn
                * (w / um_c) ** coxfit.Cf_pw * (s / um_c) ** coxfit.Cf_ps * d_avg)
    C_ox = C_A_term + C_P_term
    # --- C_s = C_underpass + C_interturn, BOTH the paper's exact closed forms, GP-legalized.
    # Underpass: N-1 crossovers (builder draws range(N-1)), each a w^2 via-oxide overlap at the
    # (N-j)/N ramp voltage -> C_up = k3·w²·(N-1)(2N-1)/(6N). Legalized 0.134·n^1.348 (fit the
    # (N-1)(2N-1)/(6N) weight over n in [3,9], -2..+6% per-N).
    # Interturn: paper's C_it = (N-1)/(6N)·C_l·8tan(pi/8)·[(2N-1)(d_out-w)-(N²-N+1)(w+s)], with
    # d_out eliminated via d_avg+n(w+s)=d_out -> EXACT sum of three N-only-weighted terms (see
    # fringing.fit_interturn_paper). A PRIOR version of this bracket was fit at 2x the correct
    # (N-1)/(6N)-family weights (an unresolved /(3N)-vs-/(6N) bookkeeping slip -- confirmed
    # algebraically and against a Palace domain-energy C_ox/C_it de-embed, see
    # COX_ONCOIL_HANDOFF.md) while C_l was never correspondingly doubled, giving a ~2x C_it
    # over-prediction. Fixed here: legalized DIRECTLY from the paper's exact form, delta_coeff=
    # 0.36 (the paper's own fit) as the only empirical constant.
    tan8_8      = 8.0 * math.tan(math.pi / 8.0)
    Cl_A_c      = Variable("Cl_A",     inter.Cl_A,     "F/m", constant=True)
    g_davg_A_c  = Variable("g_davg_A", inter.g_davg_A, "-",   constant=True)
    g_w_A_c     = Variable("g_w_A",    inter.g_w_A,    "-",   constant=True)
    g_s_A_c     = Variable("g_s_A",    inter.g_s_A,    "-",   constant=True)
    C_l_term  = Cl_A_c * (w / um_c) ** inter.Cl_pw * (s / um_c) ** inter.Cl_ps
    C_it_term = tan8_8 * C_l_term * (
        g_davg_A_c * n ** inter.g_davg_p * d_avg
      + g_w_A_c    * n ** inter.g_w_p    * w
      + g_s_A_c    * n ** inter.g_s_p    * s )
    C_s = K3_c * 0.134 * n**1.348 * w**2 + C_it_term
    # C_si (substrate self-cap, Si -> backside ground): plate-anchored Green's-function
    # closure (paper's "Substrate Capacitance"): rim self-cap (a_self=sqrt(A_foot/pi), A_foot
    # the exact octagon footprint area -> EXACT monomial in d_out) + under-metal spreading
    # plate (EXACT monomial in d_avg*n*w = l*w). BETA_CSI is the ONE EM-anchored constant
    # (coil-anchored on the C_ox-pinned refit, only 7% above the independent coil-free plate
    # anchor -- cross-validates the form, scripts/palace/close_csi_rsi.py).
    a_self_coef = math.sqrt(math.sqrt(2.0) / (2.0 * math.pi))   # a_self = a_self_coef * d_out
    Beta_csi_c = Variable("Beta_csi", BETA_CSI, "F/m", constant=True)
    e_sub_over_tsub_c = Variable("e_sub_over_tsub", e_sub_val / t_sub_val, "F/m^2",
                                 constant=True)
    C_si = Beta_csi_c * a_self_coef * d_out + e_sub_over_tsub_c * l * w
    # R_si: plate-anchored per-term R-C duality is a 2-term posynomial reciprocal (not
    # GP-legal) -- legalized as a 2D monomial in (d_out, A_metal=l*w) via rsifit above.
    Rsi_A_c = Variable("Rsi_A", rsifit.Rsi_A, "ohm", constant=True)
    um2_c = Variable("um2", 1e-12, "m^2", constant=True)
    R_si = Rsi_A_c * (d_out / um_c) ** rsifit.Rsi_pd * (l * w / um2_c) ** rsifit.Rsi_pa
    R_v  = k8 * n * (w / m_c)**(-2) * ohm_c   # notebook unit-normalisation pattern

    k_sr  = omega_sr_c / omega_c

    # -- Physical shunt via FIXED-POINT GP ------------------------------------
    # C_p (shunt cap) and R_p (shunt loss) are the Yue PARALLEL EQUIVALENT of the
    # resolved branch  C_ox -- (R_sub || C_sub):
    #   C_p = C_ox [1 + (w R_si)^2 C_si (C_ox+C_si)] / [1 + (w R_si)^2 (C_ox+C_si)^2]
    #   R_p = R_si (C_ox+C_si)^2 / C_ox^2  +  1 / (w^2 C_ox^2 R_si)
    # With the fringe C_ox (posynomial in w) and C_sub ~ d_out^1.17 these have
    # posynomial DENOMINATORS -- not GP, and not representable as gpkit signomials
    # either (1/posynomial is not a sum of monomials). They collapse to the old
    # per-area k6/k7 monomials ONLY when C_ox,C_si,R_si all scale with metal area A,
    # which they no longer do. So we FREEZE only the two DIMENSIONLESS Yue correction
    # factors and keep the geometric magnitudes live in the GP:
    #     C_p = alpha * C_ox ,   alpha = C_p_Yue / C_ox      (~0.4)
    #     R_p = kappa * R_si ,   kappa = R_p_Yue / R_si      (~2.5)
    # C_ox stays a posynomial(w) and R_si a monomial(d_out), so C_p is posynomial and
    # R_p monomial (1/R_p in the Q constraint stays GP-legal) -- the solve is a true
    # convex GP that still sees the shunt's geometry dependence (freezing the ABSOLUTE
    # C_p/R_p instead strips that dependence and makes the free-n solve degenerate).
    # We solve, recompute the Yue values at the new geometry -> refresh alpha,kappa,
    # and iterate to a damped fixed point. Operating regime x = w*tau_eff ~ 0.94..2
    # across the band, so no low-f (C_p->C_ox)/high-f (C_p->series) monomial limit is
    # safe -- hence the full conversion, refreshed each iteration.
    def _yue_factors(w_v, s_v, n_v, d_out_v, d_avg_v):
        l_v    = 8 * d_avg_v * n_v / (1 + 2**0.5)
        C_A_v  = 4.0 * tan8 * (e_ox_val / t_ox_val) * d_avg_v * n_v * (w_v + s_v)
        C_P_v  = (8.0 * tan8 * coxfit.Cf_A * n_v ** coxfit.Cf_pn
                  * (w_v * 1e6) ** coxfit.Cf_pw * (s_v * 1e6) ** coxfit.Cf_ps * d_avg_v)
        C_ox_v = C_A_v + C_P_v
        C_si_v = BETA_CSI * a_self_coef * d_out_v + (e_sub_val / t_sub_val) * l_v * w_v
        R_si_v = (rsifit.Rsi_A * (d_out_v * 1e6) ** rsifit.Rsi_pd
                  * (l_v * w_v * 1e12) ** rsifit.Rsi_pa)
        wR_v   = omega_val * R_si_v
        C_p_v  = C_ox_v * (1 + wR_v**2 * C_si_v*(C_ox_v + C_si_v)) \
                        / (1 + wR_v**2 * (C_ox_v + C_si_v)**2)
        R_p_v  = R_si_v * (C_ox_v + C_si_v)**2 / C_ox_v**2 \
                        + 1 / (omega_val**2 * C_ox_v**2 * R_si_v)
        return C_p_v / C_ox_v, R_p_v / R_si_v          # alpha, kappa

    alpha_val, kappa_val = 0.45, 2.5      # dimensionless seeds; the fixed point refines
    sol = None
    for _ in range(40):
        alpha_c = Variable("alpha", alpha_val, "-", constant=True)
        kappa_c = Variable("kappa", kappa_val, "-", constant=True)
        C_p   = alpha_c * C_ox
        R_p   = kappa_c * R_si
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
            # SRF floor: Hershenson/Mohan/Boyd/Lee eq. 13, omega_sr,min^2*Ls*Ctot_bar +
            # Rs^2*Ctot_bar/Ls <= 1, with their own two-port convention Ctot_bar = Ctot/2 (one-
            # port: Ctot_bar = Ctot -- stated just above their eq. 11). Our gamma/delta already
            # ARE that bar-substitution (gamma=omega^2*L*Ctot/cs_factor, delta=Rs^2*Ctot/cs_
            # factor/L), so substituting omega->omega_sr,min reduces the paper's constraint to
            # exactly k_sr^2*gamma + delta <= 1 -- no extra 1/2 anywhere. The originally-shipped
            # k_sr**2*gamma/2 + delta/2 <= 1 had a spurious extra /2 on top of the bar-
            # substitution, so it only guaranteed omega_sr_target/sqrt(2) for the differential
            # (cs_factor=2) topology this project actually uses -- confirmed empirically
            # (achieved SRF sat at requested/sqrt(2) before this fix).
            k_sr**2 * gamma + delta <= 1,
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
        alpha_new, kappa_new = _yue_factors(w_v, s_v, n_v, do_v, da_v)
        converged = (abs(alpha_new - alpha_val) <= 1e-3*alpha_val and
                     abs(kappa_new - kappa_val) <= 1e-3*kappa_val)
        alpha_val = 0.5*alpha_new + 0.5*alpha_val
        kappa_val = 0.5*kappa_new + 0.5*kappa_val
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
