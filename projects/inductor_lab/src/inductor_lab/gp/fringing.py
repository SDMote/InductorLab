"""Hasegawa/Schneider quasi-static fringing model for the shunt capacitance
of a tightly-bundled multi-turn spiral (see FringingMethodology.md).

The naive parallel-plate shunt cap (C = eps*A/d, footprint area l*w) ignores
fringing and under-predicts the true substrate coupling several-fold, because
the field balloons into the thick Si/SiO2 stack instead of dropping straight
down.  Following Hasegawa, the N turns -- packed close and coupling in the even
mode -- are treated as ONE wide microstrip of bundle width

    W_bun = n*w + (n-1)*s

running one loop perimeter (l_bundle = 3.3137*d_avg = l/n, NOT the full n-turn
wire length).  The oxide and substrate fringing caps are found from Schneider's
air-line impedance Z_air and combined in series (Yue high-frequency limit).

Per unit bundle length this is a *pure function of W_bun* (l_bundle cancels, and
Z_air depends only on W_bun/h), so a single monomial

    C'_ser [fF/um] = A * (W_bun[um])**b

captures it to <2% over the realistic bundle-width range.  That monomial IS
GP-legal; the transcendental Z_air (logs, additive constants, reciprocal
posynomial) and the series combination are not, which is why they are collapsed
into the fit here rather than embedded in the program.
"""

import math
from dataclasses import dataclass

_C_LIGHT = 3.0e8      # m/s (speed of light; matches FringingMethodology.md)
_EPS_0 = 8.854187e-12  # F/m (vacuum permittivity)


def _z_air(W: float, h: float) -> float:
    """Schneider quasi-static air-line impedance [ohm] for a strip of width W
    over a ground plane at distance h.  Branches on the narrow/wide regime."""
    r = W / h
    if r <= 1.0:  # narrow strip
        return 60.0 * math.log(8.0 * h / W + W / (4.0 * h))
    # wide strip
    return 120.0 * math.pi / (r + 1.393 + 0.667 * math.log(r + 1.444))


def _cprime(W: float, eps_r: float, h: float) -> float:
    """True fringing capacitance per unit length [F/m] of a strip of width W in
    a dielectric eps_r over a ground plane at distance h."""
    return eps_r / (_C_LIGHT * _z_air(W, h))


def series_cap_per_length(W: float, pdk) -> float:
    """Series (oxide || substrate) fringing shunt capacitance per unit bundle
    length [F/m] for bundle width W [m].  Pure function of W."""
    c_si = _cprime(W, pdk.eps_r_sub, pdk.t_sub)
    c_ox = _cprime(W, pdk.eps_r_ox, pdk.t_ox)
    return c_si * c_ox / (c_si + c_ox)


@dataclass(frozen=True)
class ShuntCapCoefficients:
    """Monomial fit of the bundled Hasegawa fringing shunt capacitance,

        C'_ser [fF/um] = A * (W_bun[um])**b       (per unit bundle length)

    so the GP shunt cap is  C_p = A * (W_bun/um)**b * (l_bundle/um) * fF, with
    l_bundle = l/n = 3.3137*d_avg.  Valid for W_bun in [w_lo_um, w_hi_um]."""
    A: float          # coefficient, C' in fF/um at W_bun = 1 um
    b: float          # W_bun exponent (dimensionless)
    w_lo_um: float    # fit validity range (bundle width) [um]
    w_hi_um: float
    max_err: float    # max relative fit error over the range


@dataclass(frozen=True)
class LayerCapCoefficients:
    """The oxide and substrate fringing shunt caps kept SEPARATE (not pre-combined into
    the high-frequency series), each a per-bundle-length monomial

        C'_ox[fF/um] = A_ox * (W_bun[um])**b_ox
        C'_si[fF/um] = A_si * (W_bun[um])**b_si

    Feeding these two flat endpoints into the Yue pi-network Cp_yue() reconstructs the
    frequency-dependent C_p for ANY operating frequency and PDK -- unlike the series
    collapse (C_ox||C_si), which silently assumes the operating point sits above the
    substrate relaxation crossover. That assumption is PDK/freq-specific, so we keep the
    two caps separate and let the network + omega decide the regime."""
    A_ox: float
    b_ox: float
    A_si: float
    b_si: float


# EM-calibrated for the SG13G2 backside fixture (t_ox=11.23um, t_sub=180um): the C_ox/C_si
# monomials were REFIT to the C_ox/C_si extracted (pi-model fit of Y_sub(w)) from the 20-coil
# Palace backside sweep -- residual ~+/-10% with no size bias. The raw analytic Schneider
# exponents (~0.72 ox / ~0.41 si) run too steep for wide multi-turn bundles (the single-strip
# idealization overstates width scaling; inner turns couple less), so EM pulls them to ~0.505 /
# ~0.186. Re-derive (analytic fit_layer_cap + EM trim) for a different process.
SG13G2_EM_LAYERS = LayerCapCoefficients(A_ox=0.0401, b_ox=0.505, A_si=0.0947, b_si=0.186)


def Cp_yue(C_ox: float, C_si: float, R_si: float, omega: float) -> float:
    """Yue pi-network frequency-dependent parasitic capacitance from the two flat
    (Hasegawa) endpoints and the substrate resistance:
        Y_sub = 1 / ( 1/(jw*C_ox) + 1/(1/R_si + jw*C_si) ),   C_p(w) = Im(Y_sub)/w.
    Closed form:  C_p = (C_ox + x^2 * C_ox||C_si) / (1 + x^2),  x = w*R_si*(C_ox+C_si).
    Low w -> C_ox (substrate a shifted ground); high w -> series C_ox||C_si; the roll-off
    sits at the dielectric relaxation-scaled crossover w ~ 1/(R_si*(C_ox+C_si))."""
    x = omega * R_si * (C_ox + C_si)
    c_ser = C_ox * C_si / (C_ox + C_si)
    return (C_ox + x * x * c_ser) / (1.0 + x * x)


def fit_cp_at_freq(pdk, omega: float, layers: LayerCapCoefficients = SG13G2_EM_LAYERS,
                   w_lo_um: float = 30.0, w_hi_um: float = 160.0,
                   npts: int = 80) -> ShuntCapCoefficients:
    """Collapse the frequency-dependent Yue C_p(omega) into ONE GP-legal per-length
    monomial  C'_p[fF/um] = A*(W_bun[um])**b  over the bundle-width range, evaluated AT
    `omega`. Uses the PER-PORT (pi-model split /2) oxide/substrate caps and
    R_si = eps_sub/(sigma*C_si) (the substrate dielectric relaxation, geometry-independent).
    The l_bundle factor cancels out of the transition (R_si*C_si = eps/sigma, length-free),
    so C_p = C'_p*l_bundle stays a clean monomial. PDK/frequency-general: unlike the series
    collapse it evaluates the actual regime at omega, so it's correct below, at, or above the
    crossover. Returns a ShuntCapCoefficients so design code keeps C_p = A*W_bun^b*l_bundle."""
    import numpy as np
    W = np.linspace(w_lo_um, w_hi_um, npts)                    # um
    esub = pdk.eps_r_sub * _EPS_0                              # F/m
    cp = np.empty_like(W)
    for i, w in enumerate(W):
        c_ox = layers.A_ox * w**layers.b_ox / 2.0 * 1e-9       # per-port, fF/um -> F/m
        c_si = layers.A_si * w**layers.b_si / 2.0 * 1e-9
        r_si = esub / (pdk.sigma_sub * c_si)                   # ohm*m; R_si*C_si = eps/sigma
        cp[i] = Cp_yue(c_ox, c_si, r_si, omega) * 1e9          # F/m -> fF/um
    b, ln_a = np.polyfit(np.log(W), np.log(cp), 1)
    a = math.exp(ln_a)
    max_err = float(np.max(np.abs(a * W**b - cp) / cp))
    return ShuntCapCoefficients(A=a, b=float(b), w_lo_um=w_lo_um, w_hi_um=w_hi_um,
                                max_err=max_err)


def fit_shunt_cap(pdk, w_lo_um: float = 30.0, w_hi_um: float = 160.0,
                  npts: int = 80) -> ShuntCapCoefficients:
    """Fit C'_ser[fF/um] = A*(W_bun[um])**b to the Hasegawa model over the
    bundle-width range [w_lo_um, w_hi_um].  The upper bound is clamped just
    below the substrate thickness so the fit stays within the narrow-strip
    substrate branch (a branch switch at W_bun = t_sub would kink the curve)."""
    import numpy as np

    # keep the whole range in the substrate narrow-strip branch (W/h_si <= 1)
    w_hi_um = min(w_hi_um, 0.95 * pdk.t_sub * 1e6)
    if w_hi_um <= w_lo_um:
        raise ValueError(f"empty fit range: w_lo={w_lo_um} w_hi={w_hi_um} um "
                         f"(t_sub={pdk.t_sub*1e6:.0f} um too thin?)")

    W = np.linspace(w_lo_um, w_hi_um, npts) * 1e-6            # m
    # F/m -> fF/um:  1 F/m = 1e15 fF / 1e6 um = 1e9 fF/um
    y = np.array([series_cap_per_length(w, pdk) for w in W]) * 1e9
    x = W * 1e6                                               # um
    b, ln_a = np.polyfit(np.log(x), np.log(y), 1)
    a = math.exp(ln_a)
    max_err = float(np.max(np.abs(a * x**b - y) / y))
    return ShuntCapCoefficients(A=a, b=float(b), w_lo_um=w_lo_um,
                                w_hi_um=w_hi_um, max_err=max_err)


# -- Inter-turn (turn-to-turn) series capacitance -----------------------------
# Exact closed-form from the paper's linear-differential-voltage-profile derivation (Razavi-style
# energy sum over the N-1 turn-to-turn gaps; see the paper's Sec. "Inter-turn Capacitance"):
#
#   C_it = (N-1)/(6N) * C_l * 8*tan(pi/8) * [(2N-1)(d_out-w) - (N^2-N+1)(w+s)]
#   C_l  = eps_ox*(t_m/s + delta),   delta = delta_coeff * w/(w+s)   (fill-factor fringe)
#
# delta_coeff = 0.36 is the paper's own value (fit across a broad Palace design-domain sweep,
# median 3% / RMS 14% error) -- the ONLY empirical constant in this model; the rest is exact
# geometry/energy bookkeeping, not curve-fit. NO Schneider/interleaved-1/3 model here (that was
# an earlier, superseded approximation -- see git history / FringingMethodology.md).
_LOOP_PER_DAVG = 8.0 / (1.0 + math.sqrt(2.0))     # 3.3137: one octagon loop / d_avg
DELTA_INTERTURN = 0.36                             # paper's fitted fringe coefficient


def c_l_interturn(pdk, w: float, s: float, delta_coeff: float = DELTA_INTERTURN) -> float:
    """Per-length odd-mode capacitance [F/m] of the paper's inter-turn model: C_l = eps_ox*(t_m/s
    + delta_coeff*w/(w+s)). t_m is the top-metal thickness (the facing-sidewall height)."""
    t_m = pdk.top_metals[0].thickness
    delta = delta_coeff * w / (w + s)
    return pdk.eps_r_ox * _EPS_0 * (t_m / s + delta)


def c_it(pdk, n: float, w: float, s: float, d_out: float,
        delta_coeff: float = DELTA_INTERTURN) -> float:
    """Exact closed-form inter-turn capacitance [F] for one coil (paper Eq. Ceq_inter). Physical
    model -- evaluate directly (e.g. for the pi model); fit_interturn_paper collapses it to
    GP-legal monomials."""
    tan8 = math.tan(math.pi / 8.0)
    C_l = c_l_interturn(pdk, w, s, delta_coeff)
    bracket = (2.0 * n - 1.0) * (d_out - w) - (n * n - n + 1.0) * (w + s)
    return (n - 1.0) / (6.0 * n) * C_l * 8.0 * tan8 * bracket


def c_up(pdk, n: float, w: float) -> float:
    """Exact closed-form underpass capacitance [F] for one coil (paper Eq. C_up), N-1 crossovers
    each a w^2 via-oxide overlap at the linear ramp voltage."""
    t_via = pdk.top_vias[0].thickness
    return (n - 1.0) * (2.0 * n - 1.0) / (6.0 * n) * (pdk.eps_r_ox * _EPS_0 / t_via) * w * w


# -- First-principles C_l: coupled-microstrip conformal mapping (Phase 1) -----
# Ported VERBATIM from scripts/palace/coupled_microstrip.py (verified there: Cf' convergence to
# the isolated-line limit as s->inf; Cga/Cgd checked line-by-line against the NTU EMC Chp4
# coupled-lines slides, Prof. T.L. Wu; eps_eff_yoon_kim checked <1% vs Hammerstad-Jensen; both
# eps_eq_2layer limiting cases (h1->0 => eps_r2, h1->h_total => eps_r1) verified numerically).
# Do NOT re-derive/retype these from memory when maintaining -- copy from the source script,
# which stays the single verified reference, and re-run its own __main__ checks after any change.
#
# Replaces the paper's fitted delta_coeff*w/(w+s) fringe term with the coupled-microstrip odd-
# mode gap capacitance (Cga+Cgd-Cf')/2, derived (not fit) via conformal mapping, ADDED to the
# exact sidewall term eps_ox*t_m/s (the facing-metal-thickness parallel-plate cap the classic
# zero-thickness theory has no term for at all). Validated on the 5 coil_cox coils: this fully
# derived C_l over-predicts EM by a REMARKABLY CONSISTENT ~1.3x on the 3 reliable (sparse, low-
# fill) coils (ratio 0.66, 0.68, 0.68 -- spread 3%), pointing at a missing multi-conductor/real-
# coil geometry effect (an isolated 2-conductor cross-section vs. a real coil's many-neighbor,
# curved-perimeter gap) not yet closed analytically. SCALE_INTERTURN is that ONE remaining
# constant -- a single overall scale on an already-derived SHAPE, not a fitted functional form
# (contrast the paper's delta_coeff, which fits BOTH a coefficient and the w/(w+s) shape).
def _kk_ratio(k: float) -> float:
    """K(k)/K'(k), numerically stable for k->0 (Hilberg's approximation; small-k branch uses the
    asymptotic K(k)~pi/2, K'(k)~ln(4/k) to avoid a sqrt(1-k^2)->1 precision loss)."""
    if k < 1e-6:
        return (math.pi / 2.0) / math.log(4.0 / k)
    if k <= 1.0 / math.sqrt(2.0):
        kp = math.sqrt(1.0 - k * k)
        return math.pi / math.log(2.0 * (1.0 + math.sqrt(kp)) / (1.0 - math.sqrt(kp)))
    return (1.0 / math.pi) * math.log(2.0 * (1.0 + math.sqrt(k)) / (1.0 - math.sqrt(k)))


def _c_air_partial(w: float, h: float) -> float:
    """Yoon-Kim Eq 2: air-filled isolated-line capacitance [F/m] (== 1/(Za*c))."""
    return 1.0 / (_C_LIGHT * _z_air(w, h))


def _c2_partial(w: float, h: float) -> float:
    """Yoon-Kim Eq 4: air-filled strip-to-ground(at height h) partial capacitance [F/m]."""
    k = 1.0 / math.cosh(math.pi * w / (4.0 * h))
    return 2.0 * _EPS_0 / _kk_ratio(k)      # = 2*eps0*K'(k)/K(k)


def _eps_eq_2layer(w: float, h1: float, eps_r1: float, h_total: float, eps_r2: float) -> float:
    """Yoon-Kim Eq 6+7: series combination of 2 dielectric layers stacked toward a boundary --
    h1/eps_r1 = the layer nearest the strip, h_total = h1+h2 (both layers), eps_r2 = the far
    layer. Purely geometric d1,d2 (independent of eps values) -> can be NESTED for >2 layers by
    feeding an already-combined eps_eq back in as eps_r2 (or eps_r1) of an outer application."""
    k1 = 1.0 / math.cosh(math.pi * w / (4.0 * h1))
    k = 1.0 / math.cosh(math.pi * w / (4.0 * h_total))
    d1 = _kk_ratio(k1)
    d2 = _kk_ratio(k) - d1
    return (d1 + d2) / (d1 / eps_r1 + d2 / eps_r2)


# Above-TopMetal2 cover stack (resources/SG13G2_200um.xml, z-boundaries relative to TopMetal2
# top at z=14.2303): SiO2 to z=15.7303 (1.5um more oxide), then Passive/nitride to z=16.1303
# (0.4um, eps=6.6), then the AIR dielectric block (200um, treated as the "far" reference -- a
# genuinely infinite air half-space isn't representable in this 2-layer-to-boundary formalism,
# so the declared 200um AIR block thickness is used as the outer boundary).
_T_OXIDE_ABOVE = 1.5e-6
_T_NITRIDE = 0.4e-6
_EPS_NITRIDE = 6.6
_T_AIR_BLOCK = 200e-6


def _eps_above_stack(w: float, eps_ox: float) -> float:
    """Nested Yoon-Kim 2-layer combination of the oxide(1.5um)/nitride(0.4um)/air(200um) stack
    above TopMetal2, giving one effective eps_above_eff for use in _eps_eff_full()."""
    eps_nitride_air = _eps_eq_2layer(w, _T_NITRIDE, _EPS_NITRIDE,
                                     _T_NITRIDE + _T_AIR_BLOCK, 1.0)
    h_total = _T_OXIDE_ABOVE + _T_NITRIDE + _T_AIR_BLOCK
    return _eps_eq_2layer(w, _T_OXIDE_ABOVE, eps_ox, h_total, eps_nitride_air)


def _eps_eff_full(w: float, h_below: float, eps_below: float, eps_above: float) -> float:
    """Generalized Yoon-Kim eps_eff: BOTH the below-strip region (eps_below, to the real ground
    at h_below) and the above-strip region (eps_above, an effective medium from
    _eps_above_stack) get their own dielectric enhancement -- reduces exactly to the paper's
    published single-layer Eq 5 when eps_above=1 (verified <1% vs Hammerstad-Jensen)."""
    C2 = _c2_partial(w, h_below)
    Ca = _c_air_partial(w, h_below)
    frac_below = C2 / Ca
    return 1.0 + (eps_below - 1.0) * frac_below + (eps_above - 1.0) * (1.0 - frac_below)


def _c_isolated_mixed(w: float, h: float, eps_r: float) -> float:
    """Isolated-microstrip total capacitance to ground [F/m], WITH the passivation-stack
    correction above the strip folded into _eps_eff_full."""
    Ca = _c_air_partial(w, h)
    eps_above = _eps_above_stack(w, eps_r)
    return _eps_eff_full(w, h, eps_r, eps_above) * Ca


def even_odd_cap(w: float, s: float, h: float, eps_r: float) -> tuple:
    """Per-unit-length (C_even, C_odd) [F/m] for two coplanar microstrip lines of width w, gap s,
    height h above ground, in a dielectric eps_r (below; passivation-corrected mixed dielectric
    above). Zero-thickness idealization (t not used -- see c_l_interturn_derived for the sidewall
    correction)."""
    Cp = _EPS_0 * eps_r * w / h                        # parallel-plate part (one strip)
    C_iso = _c_isolated_mixed(w, h, eps_r)
    Cf = 0.5 * (C_iso - Cp)                             # single-sided isolated-line fringe

    # even mode: neighbor at the SAME potential partially cancels the facing fringe (NTU EMC
    # Chp4 coupled-lines slides, Prof. T.L. Wu -- verified source)
    A = math.exp(-0.1 * math.exp(2.33 - 2.53 * w / h))
    Cf_prime = Cf / (1.0 + A * (h / s) * math.tanh(8.0 * s / h))
    C_even = Cp + Cf + Cf_prime

    # odd mode: extra gap capacitances (air-gap conformal + dielectric-gap)
    k = (s / h) / (s / h + 2.0 * w / h)
    Cga = _EPS_0 / _kk_ratio(k)        # _kk_ratio(k) = K(k)/K'(k), so Cga = eps0*K'(k)/K(k)
    Cgd = (_EPS_0 * eps_r / math.pi) * math.log(1.0 / math.tanh(math.pi * s / (4.0 * h))) \
        + 0.65 * Cf * (0.02 * (h / s) * math.sqrt(eps_r) + 1.0 - 1.0 / eps_r ** 2)
    C_odd = Cp + Cf + Cga + Cgd

    return C_even, C_odd


SCALE_INTERTURN = 0.68   # the ONE remaining fitted constant: median EM/derived on the 3 reliable
                         # (sparse, low-fill) coil_cox coils (0.66, 0.68, 0.68 -- 3% spread).
                         # Physical origin identified (isolated 2-conductor theory vs. a real
                         # coil's multi-neighbor, curved-perimeter gap) but NOT yet closed
                         # analytically -- see COX_ONCOIL_HANDOFF.md / session notes. A single
                         # overall SCALE on a derived shape, not a fitted functional form.


def c_ox_derived(pdk, w: float, s: float) -> float:
    """First-principles C_ox [F/m] for ONE PORT of a symmetric/differential coil: C_even(w,s)
    (the coupled-microstrip to-ground capacitance, passivation-corrected) HALVED. The /2 is a
    structural bisection-symmetry factor -- NOT a fitted constant -- from the pi-model's own
    topology (see PiModelEquations.md: C_ox appears once per port, each port's own C_ox is half
    the physical winding's to-ground capacitance in the symmetric differential structure; the
    SAME convention the previously-shipped K2=eps_ox/(2*t_ox) encodes). Multiply by the winding
    length l = 8*d_avg*n/(1+sqrt2) for the coil's total per-port C_ox.
    Validated (this session) against EM-extracted C_ox (Palace 2-port Y11+Y12 low-freq plateau,
    compare_newcs_fullpi.py's em_Cox): RMS 4.1%/max 7.8% on the matched-fixture sweep_newcs (9
    coils); RMS 15.5%/max 36.3% on the wider, different-fixture lhs_backside (98 coils) -- worse
    only for narrow-trace/wide-gap/few-turn (N=2-3, fill<0.3) coils. No fitted constant anywhere
    in this formula (the previously-shipped model's `1.01*t_ox` fringe-width fit is retired)."""
    h = pdk.t_ox
    eps_r = pdk.eps_r_ox
    C_even, _ = even_odd_cap(w, s, h, eps_r)
    return C_even / 2.0


@dataclass(frozen=True)
class CoxDerivedCoefficients:
    """GP-legal monomial legalization of c_ox_derived(w,s): a single 2D log-log fit (same
    procedure as C_l), since C_even(w,s) is not itself a monomial (the Cf' same-potential-
    neighbor correction depends on both w and s non-separably)."""
    Cox_A: float; Cox_pw: float; Cox_ps: float   # C_ox[F/m] = Cox_A * w[um]^Cox_pw * s[um]^Cox_ps
    w_lo_um: float; w_hi_um: float
    s_lo_um: float; s_hi_um: float
    max_err: float


def fit_cox_derived(pdk, w_lo_um: float = 2.0, w_hi_um: float = 28.0,
                    s_lo_um: float = 2.0, s_hi_um: float = 18.0,
                    npts: int = 30) -> CoxDerivedCoefficients:
    """Legalize c_ox_derived into a single GP-legal monomial in (w,s)."""
    import numpy as np

    W = np.linspace(w_lo_um, w_hi_um, npts) * 1e-6
    S = np.linspace(s_lo_um, s_hi_um, npts) * 1e-6
    Wg, Sg = np.meshgrid(W, S)
    Cox = np.array([[c_ox_derived(pdk, w, s) for w in W] for s in S])
    X = np.column_stack([np.ones(Wg.size), np.log(Wg.ravel() * 1e6), np.log(Sg.ravel() * 1e6)])
    coef, *_ = np.linalg.lstsq(X, np.log(Cox.ravel()), rcond=None)
    Cox_A, Cox_pw, Cox_ps = math.exp(coef[0]), float(coef[1]), float(coef[2])
    Cox_fit = Cox_A * (Wg * 1e6) ** Cox_pw * (Sg * 1e6) ** Cox_ps
    err = float(np.max(np.abs(Cox_fit - Cox) / Cox))

    return CoxDerivedCoefficients(Cox_A=Cox_A, Cox_pw=Cox_pw, Cox_ps=Cox_ps,
                                  w_lo_um=w_lo_um, w_hi_um=w_hi_um,
                                  s_lo_um=s_lo_um, s_hi_um=s_hi_um, max_err=err)


def c_l_interturn_derived(pdk, w: float, s: float,
                          scale: float = SCALE_INTERTURN) -> float:
    """First-principles C_l [F/m]: exact sidewall term (eps_ox*t_m/s, the facing-metal-thickness
    parallel-plate cap the zero-thickness theory misses) + derived conformal-mapping odd-mode
    fringe (Cga+Cgd-Cf')/2, scaled by the single residual constant above. Replaces
    c_l_interturn's fitted delta_coeff*w/(w+s) term entirely -- the w,s DEPENDENCE is now fully
    derived; `scale` is the only remaining number not obtained from geometry+conformal mapping."""
    t_m = pdk.top_metals[0].thickness
    h = pdk.t_ox
    eps_r = pdk.eps_r_ox
    C_even, C_odd = even_odd_cap(w, s, h, eps_r)
    Cm_classic = (C_odd - C_even) / 2.0
    sidewall = eps_r * _EPS_0 * t_m / s
    return scale * (sidewall + Cm_classic)


@dataclass(frozen=True)
class InterTurnPaperCoefficients:
    """GP-legal monomial legalization of the paper's EXACT inter-turn closed form (c_it above).

    d_out is eliminated via the design constraint d_avg + n*(w+s) = d_out (tight at the GP
    optimum -- same substitution used elsewhere in the GP), which turns the bracket into an EXACT
    sum of three N-only-weighted terms (no approximation in N, w, s, d_avg -- only C_l(w,s) and
    the three pure N power laws need fitting, since gpkit posynomials require single power-law
    monomials, not the raw linear-in-N subtractions):

        C_it = 8*tan(pi/8) * C_l(w,s) * [ g_davg(n)*d_avg + g_w(n)*w + g_s(n)*s ]
        g_davg(n) = (n-1)*(2n-1)/(6n)      g_w(n) = (n-1)*(n-2)/6      g_s(n) = (n-1)**2*(n+1)/(6n)

    C_l(w,s) needs a small 2D log-log fit (delta_coeff*w/(w+s) is not itself a monomial); each
    g_x(n) is a clean 1D power-law fit (g_w is exactly 0 at n=2, so n_lo must stay > 2). ALL fits
    are to the closed form, not to EM data -- delta_coeff is the one open physical parameter."""
    Cl_A: float; Cl_pw: float; Cl_ps: float          # C_l[F/m] = Cl_A * w[um]^Cl_pw * s[um]^Cl_ps
    g_davg_A: float; g_davg_p: float                 # g_davg(n) = g_davg_A * n^g_davg_p
    g_w_A: float; g_w_p: float
    g_s_A: float; g_s_p: float
    n_lo: float; n_hi: float
    w_lo_um: float; w_hi_um: float
    s_lo_um: float; s_hi_um: float
    delta_coeff: float
    max_err: float    # worst-case relative fit error over the box (all four fits combined)


def fit_interturn_paper(pdk, n_lo: float = 3.0, n_hi: float = 9.0,
                        w_lo_um: float = 2.0, w_hi_um: float = 28.0,
                        s_lo_um: float = 2.0, s_hi_um: float = 18.0,
                        delta_coeff: float = DELTA_INTERTURN,
                        npts: int = 30) -> InterTurnPaperCoefficients:
    """Legalize the paper's exact inter-turn closed form into GP-legal monomials, per
    InterTurnPaperCoefficients' docstring. NO EM calibration in the geometry fit -- only
    delta_coeff is an external (paper-fit) physical parameter."""
    import numpy as np

    # (1) C_l(w,s): 2D log-log least-squares (not separable due to the w/(w+s) fringe term)
    W = np.linspace(w_lo_um, w_hi_um, npts) * 1e-6
    S = np.linspace(s_lo_um, s_hi_um, npts) * 1e-6
    Wg, Sg = np.meshgrid(W, S)
    Cl = np.array([[c_l_interturn(pdk, w, s, delta_coeff) for w in W] for s in S])
    X = np.column_stack([np.ones(Wg.size), np.log(Wg.ravel() * 1e6), np.log(Sg.ravel() * 1e6)])
    coef, *_ = np.linalg.lstsq(X, np.log(Cl.ravel()), rcond=None)
    Cl_A, Cl_pw, Cl_ps = math.exp(coef[0]), float(coef[1]), float(coef[2])
    Cl_fit = Cl_A * (Wg * 1e6) ** Cl_pw * (Sg * 1e6) ** Cl_ps
    err_cl = float(np.max(np.abs(Cl_fit - Cl) / Cl))

    # (2) three EXACT N-only weights -> clean power-law monomial fits (no subtraction left)
    Nn = np.linspace(n_lo, n_hi, npts)
    g_davg = (Nn - 1.0) * (2.0 * Nn - 1.0) / (6.0 * Nn)
    g_w    = (Nn - 1.0) * (Nn - 2.0) / 6.0
    g_s    = (Nn - 1.0) ** 2 * (Nn + 1.0) / (6.0 * Nn)

    def _fit_power(y):
        p, ln_a = np.polyfit(np.log(Nn), np.log(y), 1)
        a = math.exp(ln_a)
        return a, float(p), float(np.max(np.abs(a * Nn**p - y) / y))

    g_davg_A, g_davg_p, err_davg = _fit_power(g_davg)
    g_w_A, g_w_p, err_w = _fit_power(g_w)
    g_s_A, g_s_p, err_s = _fit_power(g_s)

    return InterTurnPaperCoefficients(
        Cl_A=Cl_A, Cl_pw=Cl_pw, Cl_ps=Cl_ps,
        g_davg_A=g_davg_A, g_davg_p=g_davg_p, g_w_A=g_w_A, g_w_p=g_w_p, g_s_A=g_s_A, g_s_p=g_s_p,
        n_lo=n_lo, n_hi=n_hi, w_lo_um=w_lo_um, w_hi_um=w_hi_um, s_lo_um=s_lo_um, s_hi_um=s_hi_um,
        delta_coeff=delta_coeff, max_err=max(err_cl, err_davg, err_w, err_s))


@dataclass(frozen=True)
class InterTurnDerivedCoefficients:
    """Same GP legalization as InterTurnPaperCoefficients (identical N-weight fits g_davg/g_w/g_s
    -- those depend only on the exact geometric bracket, not on which C_l model is used), but
    C_l(w,s) is fit to c_l_interturn_derived (coupled-microstrip conformal mapping + exact
    sidewall term) instead of the paper's fitted delta_coeff*w/(w+s). `scale` (SCALE_INTERTURN)
    is the one remaining empirical constant -- an overall multiplier on an already-derived shape,
    not a fitted functional form."""
    Cl_A: float; Cl_pw: float; Cl_ps: float
    g_davg_A: float; g_davg_p: float
    g_w_A: float; g_w_p: float
    g_s_A: float; g_s_p: float
    n_lo: float; n_hi: float
    w_lo_um: float; w_hi_um: float
    s_lo_um: float; s_hi_um: float
    scale: float
    max_err: float


def fit_interturn_derived(pdk, n_lo: float = 3.0, n_hi: float = 9.0,
                          w_lo_um: float = 2.0, w_hi_um: float = 28.0,
                          s_lo_um: float = 2.0, s_hi_um: float = 18.0,
                          scale: float = SCALE_INTERTURN,
                          npts: int = 30) -> InterTurnDerivedCoefficients:
    """Legalize c_l_interturn_derived (first-principles C_l) into GP-legal monomials, per
    InterTurnDerivedCoefficients' docstring."""
    import numpy as np

    W = np.linspace(w_lo_um, w_hi_um, npts) * 1e-6
    S = np.linspace(s_lo_um, s_hi_um, npts) * 1e-6
    Wg, Sg = np.meshgrid(W, S)
    Cl = np.array([[c_l_interturn_derived(pdk, w, s, scale) for w in W] for s in S])
    X = np.column_stack([np.ones(Wg.size), np.log(Wg.ravel() * 1e6), np.log(Sg.ravel() * 1e6)])
    coef, *_ = np.linalg.lstsq(X, np.log(Cl.ravel()), rcond=None)
    Cl_A, Cl_pw, Cl_ps = math.exp(coef[0]), float(coef[1]), float(coef[2])
    Cl_fit = Cl_A * (Wg * 1e6) ** Cl_pw * (Sg * 1e6) ** Cl_ps
    err_cl = float(np.max(np.abs(Cl_fit - Cl) / Cl))

    Nn = np.linspace(n_lo, n_hi, npts)
    g_davg = (Nn - 1.0) * (2.0 * Nn - 1.0) / (6.0 * Nn)
    g_w    = (Nn - 1.0) * (Nn - 2.0) / 6.0
    g_s    = (Nn - 1.0) ** 2 * (Nn + 1.0) / (6.0 * Nn)

    def _fit_power(y):
        p, ln_a = np.polyfit(np.log(Nn), np.log(y), 1)
        a = math.exp(ln_a)
        return a, float(p), float(np.max(np.abs(a * Nn**p - y) / y))

    g_davg_A, g_davg_p, err_davg = _fit_power(g_davg)
    g_w_A, g_w_p, err_w = _fit_power(g_w)
    g_s_A, g_s_p, err_s = _fit_power(g_s)

    return InterTurnDerivedCoefficients(
        Cl_A=Cl_A, Cl_pw=Cl_pw, Cl_ps=Cl_ps,
        g_davg_A=g_davg_A, g_davg_p=g_davg_p, g_w_A=g_w_A, g_w_p=g_w_p, g_s_A=g_s_A, g_s_p=g_s_p,
        n_lo=n_lo, n_hi=n_hi, w_lo_um=w_lo_um, w_hi_um=w_hi_um, s_lo_um=s_lo_um, s_hi_um=s_hi_um,
        scale=scale, max_err=max(err_cl, err_davg, err_w, err_s))


# -- Oxide capacitance: Scuderi-style solid-disk model (paper's "Oxide Capacitance" section) --
# C_ox (per port) = (C_A + C_P)/2, Scuderi et al. 2004 (IEEE TCAS-I): C_A = parallel-plate cap
# over the WHOLE spiral footprint treated as a solid disk (area = outer octagon minus inner
# octagon), C_P = edge-fringe cap over the inner+outer winding perimeter only (bundle of N turns
# treated as ONE wide conductor of width W_bun, Hasegawa convention). Both octagon area and
# perimeter reduce EXACTLY (via d_out = d_avg + n*(w+s)) to the shared factor 8*tan(pi/8):
#     A_solid  = 8*(sqrt2-1)*d_avg*n*(w+s)             (exact octagon solid-disk area)
#     P_total  = 16*(sqrt2-1)*d_avg = 2*l/n            (exact inner+outer perimeter)
# giving, with NO fitting for the area term:
#     C_A = A_solid*eps_ox*eps0/t_ox
# The edge-fringe term C_P = Cf(W_bun)*P_total needs Cf (one isolated-bundle-edge fringe cap
# per unit length, from the SAME Hammerstad/Yoon-Kim isolated-line machinery as C_it) legalized
# as a monomial -- W_bun = n*w+(n-1)*s is a SUM, not itself a GP variable, so Cf is fit directly
# as a 3-variable (n,w,s) power law rather than evaluated as W_bun**p.
def cf_bundle_fringe(pdk, w: float, s: float, n: float) -> float:
    """One-edge isolated-bundle fringe capacitance [F/m] at bundle width W_bun=n*w+(n-1)*s,
    from the passivation-corrected Hammerstad/Yoon-Kim isolated-microstrip machinery
    (_c_isolated_mixed) already used for C_it -- the fringe PART only (parallel-plate part
    subtracted out, since that is already counted exactly in C_A)."""
    W_bun = n * w + (n - 1.0) * s
    C_iso = _c_isolated_mixed(W_bun, pdk.t_ox, pdk.eps_r_ox)
    Cp_bun = _EPS_0 * pdk.eps_r_ox * W_bun / pdk.t_ox
    return 0.5 * (C_iso - Cp_bun)


@dataclass(frozen=True)
class CoxScuderiCoefficients:
    """GP-legal monomial legalization of cf_bundle_fringe(w,s,n): a 3D log-log fit (W_bun is a
    sum of n,w,s, not separable into a single power-law variable). The solid-disk area term
    C_A needs NO fit (exact octagon identity) -- only this edge-fringe correction does.
    Full per-port C_ox = 4*tan(pi/8)*d_avg*[ eps_ox/t_ox*n*(w+s) + 2*Cf_A*n^Cf_pn*w[um]^Cf_pw*
    s[um]^Cf_ps ]. Validated (this session) vs EM (C_ox now pinned in the fixed-Cox 2-parameter
    Csi/Rsi refit, scripts/palace/close_csi_rsi.py) -- direct RMS 4.8%/10.5% on lhs_backside/
    sweep_newcs (98/9 coils, no monomial legalization yet in that check)."""
    Cf_A: float; Cf_pn: float; Cf_pw: float; Cf_ps: float
    n_lo: float; n_hi: float
    w_lo_um: float; w_hi_um: float
    s_lo_um: float; s_hi_um: float
    max_err: float


def fit_cox_scuderi(pdk, n_lo: float = 2.0, n_hi: float = 9.0,
                    w_lo_um: float = 2.0, w_hi_um: float = 28.0,
                    s_lo_um: float = 2.0, s_hi_um: float = 18.0,
                    npts: int = 12) -> CoxScuderiCoefficients:
    """Legalize cf_bundle_fringe into a single GP-legal monomial in (n,w,s)."""
    import numpy as np

    Nn = np.linspace(n_lo, n_hi, npts)
    W = np.linspace(w_lo_um, w_hi_um, npts) * 1e-6
    S = np.linspace(s_lo_um, s_hi_um, npts) * 1e-6
    Ng, Wg, Sg = np.meshgrid(Nn, W, S, indexing="ij")
    # skip unrealistic corners (large n AND large w AND large s simultaneously -- never occurs
    # for real coils, since many wide turns don't fit in a compact spiral) where W_bun overflows
    # _eps_above_stack's nested cosh (math range error above ~360um bundle width).
    Wbun = Ng * Wg + (Ng - 1.0) * Sg
    ok = Wbun * 1e6 < 300.0
    Cf = np.full(Ng.shape, np.nan)
    for i in range(Ng.shape[0]):
        for j in range(Ng.shape[1]):
            for k in range(Ng.shape[2]):
                if ok[i, j, k]:
                    Cf[i, j, k] = cf_bundle_fringe(pdk, Wg[i, j, k], Sg[i, j, k], Ng[i, j, k])
    m = ok.ravel()
    X = np.column_stack([np.ones(Ng.size), np.log(Ng.ravel()),
                         np.log(Wg.ravel() * 1e6), np.log(Sg.ravel() * 1e6)])[m]
    coef, *_ = np.linalg.lstsq(X, np.log(Cf.ravel()[m]), rcond=None)
    Cf_A = math.exp(coef[0]); Cf_pn = float(coef[1]); Cf_pw = float(coef[2]); Cf_ps = float(coef[3])
    Cf_fit = Cf_A * Ng ** Cf_pn * (Wg * 1e6) ** Cf_pw * (Sg * 1e6) ** Cf_ps
    err = float(np.max(np.abs(Cf_fit.ravel()[m] - Cf.ravel()[m]) / Cf.ravel()[m]))

    return CoxScuderiCoefficients(Cf_A=Cf_A, Cf_pn=Cf_pn, Cf_pw=Cf_pw, Cf_ps=Cf_ps,
                                  n_lo=n_lo, n_hi=n_hi, w_lo_um=w_lo_um, w_hi_um=w_hi_um,
                                  s_lo_um=s_lo_um, s_hi_um=s_hi_um, max_err=err)


# -- Substrate capacitance/resistance: plate-anchored Green's-function closure --------------
# C_si = beta*a_self + eps_si*eps0/t_sub*A_metal   (rim self-cap + under-metal spreading plate,
# scripts/palace/greens_closed.py). a_self = sqrt(A_foot/pi), A_foot = octagon footprint area
# (exact in d_out); A_metal = l*w (exact in d_avg,n,w). BOTH terms are already honest monomials
# -- NO fit needed for the FORM, only beta is an anchored constant (coil-anchored on the
# C_ox-pinned 2-parameter Csi/Rsi refit, scripts/palace/close_csi_rsi.py: 4.033e-10 F/m, only
# 7% above the independent coil-free plate anchor 3.778e-10 F/m -- cross-validates the form).
BETA_CSI = 4.033e-10        # F/m, coil-anchored (close_csi_rsi.py, lhs_backside+sweep_newcs)
TAU_SELF_COIL = 68.2e-12    # s, coil-anchored (same script). tau_si (bulk) is NOT a fitted
                            # constant -- it's eps_si*eps0/sigma_sub, a pure stackup property,
                            # computed from pdk at call time.
_A_SELF_COEF = math.sqrt(math.sqrt(2.0) / (2.0 * math.pi))   # a_self = _A_SELF_COEF * d_out


def c_si_derived(pdk, d_out: float, d_avg: float, n: float, w: float,
                 beta: float = BETA_CSI) -> float:
    """Exact closed-form substrate self-capacitance (per port): rim self-cap + under-metal
    parallel-plate spreading term. No fit -- both terms are monomials in (d_out) and
    (d_avg,n,w) respectively; only beta is an EM-anchored constant."""
    a_self = _A_SELF_COEF * d_out
    l = 8.0 * d_avg * n / (1.0 + 2.0 ** 0.5)
    Cpp = pdk.eps_r_sub * _EPS_0 * l * w / pdk.t_sub
    return beta * a_self + Cpp


def g_si_derived(pdk, d_out: float, d_avg: float, n: float, w: float,
                 beta: float = BETA_CSI, tau_self: float = TAU_SELF_COIL) -> float:
    """Substrate conductance (per port) via per-term R-C duality: G_si = Cpp/tau_si +
    Cself/tau_self. tau_si = eps_si*eps0/sigma_sub is the PURE bulk-silicon relaxation (exact,
    no fit); tau_self is the ONE coil-anchored constant (rim self-cap spreading resistance of an
    annular/spiral rim differs from a solid plate's rim -- see close_csi_rsi.py)."""
    a_self = _A_SELF_COEF * d_out
    l = 8.0 * d_avg * n / (1.0 + 2.0 ** 0.5)
    Cpp = pdk.eps_r_sub * _EPS_0 * l * w / pdk.t_sub
    Cself = beta * a_self
    tau_si = pdk.eps_r_sub * _EPS_0 / pdk.sigma_sub
    return Cpp / tau_si + Cself / tau_self


@dataclass(frozen=True)
class RsiCoefficients:
    """GP-legal power-law monomial legalization of R_si = 1/g_si_derived (a 2-term posynomial
    reciprocal, not itself GP-legal -- same non-representable-ratio issue as the Yue C_p/R_p
    rollup), as a 2D log-log fit in (d_out, A_metal=l*w) -- the two independent drivers of the
    two duality terms (rim self-cap / under-metal plate). A 1D fit vs d_out alone left ~56% max
    error (the metal-area term is NOT negligible at fixed d_out); this 2D form is still exactly
    GP-legal since both d_out and l*w are already live monomials in the GP."""
    Rsi_A: float; Rsi_pd: float; Rsi_pa: float   # R_si[ohm]=Rsi_A*d_out[um]^Rsi_pd*A[um^2]^Rsi_pa
    d_out_lo_um: float; d_out_hi_um: float
    max_err: float


def fit_rsi_derived(pdk, n_lo: float = 2.0, n_hi: float = 9.0,
                    w_lo_um: float = 2.0, w_hi_um: float = 28.0,
                    s_lo_um: float = 2.0, s_hi_um: float = 18.0,
                    davg_lo_um: float = 80.0, davg_hi_um: float = 500.0,
                    npts: int = 8, beta: float = BETA_CSI,
                    tau_self: float = TAU_SELF_COIL) -> RsiCoefficients:
    """Legalize R_si=1/g_si_derived into a single GP-legal monomial in (d_out, A_metal=l*w),
    sampled over a realistic (n,w,s,d_avg) population."""
    import numpy as np

    Nn = np.linspace(n_lo, n_hi, npts)
    W = np.linspace(w_lo_um, w_hi_um, npts) * 1e-6
    S = np.linspace(s_lo_um, s_hi_um, npts) * 1e-6
    Davg = np.linspace(davg_lo_um, davg_hi_um, npts) * 1e-6
    d_out_l, A_l, R_l = [], [], []
    for nv in Nn:
        for wv in W:
            for sv in S:
                for dav in Davg:
                    d_out = dav + nv * (wv + sv)
                    l = 8.0 * dav * nv / (1.0 + 2.0 ** 0.5)
                    g = g_si_derived(pdk, d_out, dav, nv, wv, beta, tau_self)
                    d_out_l.append(d_out)
                    A_l.append(l * wv)
                    R_l.append(1.0 / g)
    d_out_a = np.array(d_out_l); A_a = np.array(A_l); R_a = np.array(R_l)
    X = np.column_stack([np.ones_like(d_out_a), np.log(d_out_a * 1e6), np.log(A_a * 1e12)])
    coef, *_ = np.linalg.lstsq(X, np.log(R_a), rcond=None)
    a = math.exp(coef[0]); pd = float(coef[1]); pa = float(coef[2])
    Rfit = a * (d_out_a * 1e6) ** pd * (A_a * 1e12) ** pa
    err = float(np.max(np.abs(Rfit - R_a) / R_a))

    return RsiCoefficients(Rsi_A=a, Rsi_pd=pd, Rsi_pa=pa,
                           d_out_lo_um=float(d_out_a.min() * 1e6),
                           d_out_hi_um=float(d_out_a.max() * 1e6), max_err=err)
