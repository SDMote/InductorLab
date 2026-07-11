#!/usr/bin/env python3
"""
spiral_cturn.py  (standalone experiment -- does NOT modify the production GP)
============================================================================
A copy of inductor_lab.gp.spiral.design_inductor with the calibrated inter-turn
capacitance term added to C_s:

    C_s = k3*n*w^2  +  gamma * c_turn_phys * n * d_avg / s
    c_turn_phys = 8(sqrt2-1) * eps0 * eps_r * t_metal      (octagon sidewall plate)
    gamma       = 0.594                                    (fit_gamma.py, both sweeps)

C_s stays a monomial->posynomial sum, which is GP-valid because C_s only enters the
*inequality* constraints (Q, SRF). gamma=0 recovers the original C model.

Two corrections vs the production GP:
  1. inter-turn capacitance added to C_s (above);
  2. the extra /2 on the SRF constraints removed. Self-resonance is gamma+delta=1
     (cell-13 Q numerator), so the floor is  k_sr^2*gamma + delta <= 1  (Hershenson
     eq 13), not the notebook's k_sr^2*gamma/2 + delta/2 (which was 2x loose). The
     justified C_tot/2 stays inside gamma/delta.

main() re-optimises L=1..20 nH at 2.5 GHz with gamma=0 vs gamma=0.594.
"""
import csv
import importlib.util
import math
from dataclasses import replace
from pathlib import Path

from gpkit import Model, Variable

from inductor_lab.gp.spiral import InductanceCoefficients, Topology
from inductor_lab.gp.fringing import fit_cp_at_freq
from inductor_lab.pdk.sg13g2 import BACKLAPPING, SG13G2

REPO = Path(__file__).resolve().parents[2]   # scripts/palace/<file> -> repo root
_MU_0, _EPS_0 = 4 * math.pi * 1e-7, 8.854187e-12
_K_E = 1 / (1 + 2 ** 0.5)                       # octagon transition-gap factor
FREQ = 2.5e9
GAMMA = 0.68                                   # inter-turn correction, refit vs Niknejad C_BR
                                               # (was 0.594 from one-port SRF; C_BR isolates
                                               # the series cap from substrate C_p -> cleaner)

# Calibrated series (bridge) capacitance C_BR, fit DIRECTLY to Niknejad C_BR over the pooled
# lhs* sweeps (scripts/fit_cbr.py direct-monomial fit): 13.4% RMS / 5.5% median, vs ~29% for
# the gamma*n*d_avg/s form. ONE monomial replaces both the underpass w^2 term (largely
# spurious -- OVERVIEW 5.1) and the linear inter-turn term. Lengths normalised by 1 um, so the
# coefficient is plain Farads. Still a monomial in (n,d_avg,s,w) -> GP-legal in C_s.
#   C_s = CBR_COEF * n^CBR_AN * (d_avg/um)^CBR_ADAVG * (s/um)^CBR_AS * (w/um)^CBR_AW
# The point of this term: its s exponent is -0.63 (not -1) and its n exponent 1.55 (not 1), so
# the optimizer can no longer "buy" self-resonance headroom by widening s / adding turns the way
# the old model let it -- C_s now GROWS with L like Palace, so predicted SRF falls like Palace.
CBR_COEF = 3.2116e-17                           # [F]
CBR_AN, CBR_ADAVG, CBR_AS, CBR_AW = 1.5514, 0.9842, -0.6258, 0.3026

# THE canonical inductance fit: Mohan exponents + BETA re-fit to the true PEC-ground backside
# (coefficients_sym_sg13g2.py). Full d_out form (a1!=0) -> use design() / design_inductor().
_ac = REPO / "ASITIC" / "coefficients_sym_sg13g2.py"
_spec = importlib.util.spec_from_file_location("ac", _ac)
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
COEFFS = InductanceCoefficients(_m.BETA, _m.A1, _m.A2, _m.A3, _m.A4, _m.A5)
PDK = replace(SG13G2, t_sub=BACKLAPPING[200])


def _k6(k2, k4, k5, w):
    return k5 * (k2 + k4) ** 2 / k2 ** 2 + 1.0 / (k2 ** 2 * k5 * w ** 2)


def _k7(k2, k4, k5, w):
    return (k2 * (k4 * k5 ** 2 * w ** 2 * (k2 + k4) + 1.0)
            / (k5 ** 2 * w ** 2 * (k2 + k4) ** 2 + 1.0))


def design(nanohenries, pdk, coeffs=COEFFS, gamma=GAMMA, topology=Topology.DIFFERENTIAL,
           min_srf_hz=7e9, fixed_n=None, max_area_m2=None,
           w_max=None, s_max=None, davg_max=None, rho_max=0.35):
    """design_inductor + inter-turn C term. Returns dict (or raises on infeasible)."""
    cs_factor = 2 if topology is Topology.DIFFERENTIAL else 1
    omega_val, omega_sr_val = 2 * math.pi * FREQ, 2 * math.pi * min_srf_hz
    metal, via = pdk.top_metals[0], pdk.top_vias[0]
    sigma_m, t_m = metal.sigma, metal.thickness
    sigma_v, t_via, a_via, b_via = via.sigma, via.thickness, via.width, via.space
    e_ox, t_ox = pdk.eps_r_ox * _EPS_0, pdk.t_ox
    e_sub, t_sub, sigma_sub = pdk.eps_r_sub * _EPS_0, pdk.t_sub, pdk.sigma_sub

    skin_m = math.sqrt(2 / (omega_val * _MU_0 * sigma_m))
    k1 = (1.0 / (sigma_m * skin_m * (1 - math.exp(-t_m / skin_m)))
          if skin_m < t_m else 1.0 / (sigma_m * t_m))
    k2 = e_ox / (2 * t_ox); k3 = e_ox / t_via
    k4 = e_sub / (2 * t_sub); k5 = 2 * t_sub / sigma_sub
    k6 = _k6(k2, k4, k5, omega_val); k7 = _k7(k2, k4, k5, omega_val)
    skin_v = math.sqrt(2 / (omega_val * _MU_0 * sigma_v))
    k8 = (2 * t_via / (sigma_v * a_via * skin_v * (1 - math.exp(-a_via / skin_v)) * (a_via + b_via) ** 2)
          if skin_v < a_via else 2 * t_via * (a_via + b_via) ** 2 / (sigma_v * a_via ** 2))

    # the new inter-turn coefficient: Cs_turn = c_turn * n * d_avg / s
    c_turn = gamma * 8 * (math.sqrt(2) - 1) * _EPS_0 * pdk.eps_r_ox * t_m   # [F]
    # Fringing shunt cap: Yue C_p(omega) from the EM-calibrated Hasegawa oxide/substrate layers,
    # collapsed to a GP-legal per-bundle-length monomial at the design omega (fringing.py). Replaces
    # the parallel-plate C_p = K7*l*w, which under-predicted substrate coupling ~5x (SRF bug).
    shunt = fit_cp_at_freq(pdk, omega_val)

    omega_c = Variable("omega", omega_val, "rad/s", constant=True)
    omega_sr_c = Variable("omega_sr", omega_sr_val, "rad/s", constant=True)
    L_req_c = Variable("L_req", nanohenries * 1e-9, "H", constant=True)
    um_c = Variable("um", 1e-6, "m", constant=True)
    nH_c = Variable("nH", 1e-9, "H", constant=True)
    m_c = Variable("m", 1, "m", constant=True)
    ohm_c = Variable("ohm", 1, "ohm", constant=True)
    K1_c = Variable("K1", k1, "ohm", constant=True)
    K2_c = Variable("K2", k2, "F/m^2", constant=True)
    K3_c = Variable("K3", k3, "F/m^2", constant=True)
    K4_c = Variable("K4", k4, "F/m^2", constant=True)
    K5_c = Variable("K5", k5, "ohm*m^2", constant=True)
    K6_c = Variable("K6", k6, "ohm*m^2", constant=True)
    K7_c = Variable("K7", k7, "F/m^2", constant=True)
    Cturn_c = Variable("Cturn", c_turn, "F", constant=True)
    Acap_c = Variable("Acap", shunt.A, "-", constant=True)
    fF_c = Variable("fF", 1e-15, "F", constant=True)
    w_min_c = Variable("w_min", metal.w_min, "m", constant=True)
    s_min_c = Variable("s_min", metal.s_min, "m", constant=True)

    d_out = Variable("d_out", "m", positive=True)
    d_avg = Variable("d_avg", "m", positive=True)
    w = Variable("w", "m", positive=True)
    s = Variable("s", "m", positive=True)
    n = Variable("n", "-", positive=True)
    W_bun = Variable("W_bun", "m", positive=True)   # bundle width n*w+(n-1)*s
    Q_min = Variable("Q_min", "-", positive=True)
    R_s = Variable("R_s", "ohm", positive=True)

    l = 8 * d_avg * n / (1 + 2 ** 0.5)
    L = (coeffs.beta * (d_out / um_c) ** coeffs.a1 * (w / um_c) ** coeffs.a2
         * (d_avg / um_c) ** coeffs.a3 * n ** coeffs.a4 * (s / um_c) ** coeffs.a5 * nH_c)
    R_m = K1_c * l / w
    # C_s = underpass (Yue) + inter-turn. The notebook's differential derivation has the
    # series cap driven by the FULL differential V (the Ys*V term), and the symmetric coil
    # is interleaved (adjacent turns are opposite halves, +-V/2), so the inter-turn cap
    # sees ~full V and adds straight into C_s -- no single-ended (n-1)/n^2 reduction.
    C_s = K3_c * n * w ** 2 + Cturn_c * n * d_avg / s
    R_v = k8 * n * (w / m_c) ** (-2) * ohm_c
    R_p = K6_c / (l * w)
    l_bundle = l / n                                 # one loop perimeter = 3.3137*d_avg
    C_p = Acap_c * (W_bun / um_c) ** shunt.b * (l_bundle / um_c) * fF_c
    C_tot = C_p + cs_factor * C_s
    rho = omega_c * L / R_s
    gamma_g = omega_c ** 2 * L * C_tot / cs_factor
    delta = R_s ** 2 * C_tot / cs_factor / L
    k_sr = omega_sr_c / omega_c

    cons = [
        L == L_req_c, R_s >= R_m + R_v,
        Q_min * (cs_factor * R_p + (rho ** 2 + 1) * R_s) / (rho * cs_factor * R_p) + delta + gamma_g <= 1,
        # Self-resonance is gamma+delta=1 (cell-13 Q numerator: 2Ls = C_tot(Ls^2 w^2+Rs^2)).
        # The gamma/delta already carry the justified C_tot/2; the extra /2 the notebook
        # had on the constraints made the SRF floor 2x loose (true floor ~ w_sr_min/sqrt2).
        # Fixed below: operate below SRF, and enforce SRF >= w_sr_min per Hershenson eq (13).
        delta + gamma_g <= 1,
        k_sr ** 2 * gamma_g + delta <= 1,
        w >= w_min_c, s >= s_min_c, d_avg + n * s + n * w <= d_out,
        # Drawability: the inner opening must fit the center transition gap, else the GDS
        # self-overlaps. d_in = d_out - 2(n*w+(n-1)*s) >= e + 2w, e = w+s+(w+2s)/(1+sqrt2).
        # n-1 ~ n keeps it a posynomial (conservative). So the GP returns the optimal
        # *drawable* inductor instead of an un-meshable one.
        2 * n * w + 2 * n * s + (3 + _K_E) * w + (1 + 2 * _K_E) * s <= d_out,
        # Bundle width for the fringing C_p: exact n*w+(n-1)*s, but -s is signomial, so use
        # n*(w+s) (over-counts one gap, conservative). C_p rises with W_bun so the solver drives
        # it to this lower bound -> tight. Posynomial <= 1.
        n * w / W_bun + n * s / W_bun <= 1,
    ]
    if fixed_n is not None:
        cons.append(n == fixed_n)
    if max_area_m2 is not None:
        cons.append(d_out ** 2 <= Variable("A_max", max_area_m2, "m^2", constant=True))
    # Optional UPPER bounds to keep the design inside the L-fit's sampled box (n2-9, w2-28,
    # s2-18, d_avg60-600 um); monomial <= const, GP-legal.
    if w_max is not None:
        cons.append(w <= Variable("w_max", w_max, "m", constant=True))
    if s_max is not None:
        cons.append(s <= Variable("s_max", s_max, "m", constant=True))
    if davg_max is not None:
        cons.append(d_avg <= Variable("davg_max", davg_max, "m", constant=True))
    if rho_max is not None:
        # Fill-factor ceiling rho = (winding width)/d_avg <= rho_max. W_bun is the winding width
        # (n*w+(n-1)*s, driven tight above), so W_bun <= rho_max*d_avg. Keeps the design out of
        # the high-rho / wide-trace corner where the Mohan L monomial (current-sheet approx)
        # fails -- fit is ~5% for rho<0.35 vs a lone +53% outlier at rho~0.44. Monomial<=monomial.
        cons.append(W_bun <= rho_max * d_avg)
    sol = Model(Q_min ** -1, cons).solve(verbosity=0)

    def q(e):
        return e.sub(sol["variables"]).value
    L_val = q(L).to("H").magnitude
    Rs_val = q(R_s).to("ohm").magnitude
    Ctot_val = q(C_tot).to("F").magnitude
    fsr = math.sqrt(2 / (L_val * Ctot_val) - Rs_val ** 2 / L_val ** 2) / (2 * math.pi)
    return dict(n=float(q(n)), w=q(w).to("m").magnitude, s=q(s).to("m").magnitude,
                d_out=q(d_out).to("m").magnitude, d_avg=q(d_avg).to("m").magnitude,
                Q=float(q(Q_min)), L=L_val, Ctot=Ctot_val, f_sr=fsr)


def best_int(nanohenries, _design=design, **kw):
    """Best integer-turn design, or None if infeasible (e.g. SRF floor + area cap)."""
    try:
        cont = _design(nanohenries, PDK, **kw)
    except Exception:
        return None
    best = None
    for N in sorted({max(1, math.floor(cont["n"])), math.ceil(cont["n"])}):
        try:
            d = _design(nanohenries, PDK, fixed_n=N, **kw)
        except Exception:
            continue
        if best is None or d["Q"] > best["Q"]:
            best = d
    return best


def main():
    # measured Palace SRF by target L
    meas = {}
    cf = REPO / "scripts/.out/sweep_2p5ghz_fsweep/comparison_fsweep.csv"
    if cf.exists():
        for r in csv.DictReader(open(cf)):
            try:
                meas[int(r["target"])] = float(r["palSRF"])
            except ValueError:
                pass

    print(f"Re-optimised at 2.5 GHz, integer turns, SRF floor 7 GHz.  gamma_cturn = {GAMMA}\n")
    print(f"{'L':>3} | {'ORIGINAL (gamma=0)':^28} | {'+ Cs_turn (gamma=0.594)':^33} | {'Palace':>6}")
    print(f"{'':>3} | {'N':>2} {'w':>5} {'dout':>5} {'Q':>5} {'SRFpred':>7} | "
          f"{'N':>2} {'w':>5} {'dout':>5} {'Q':>5} {'SRFpred':>7} | {'SRF':>6}")
    print("-" * 86)
    for Lt in range(1, 21):
        try:
            o = best_int(Lt, gamma=0.0)
            os = f"{round(o['n']):>2} {o['w']*1e6:5.1f} {o['d_out']*1e6:5.0f} {o['Q']:5.1f} {o['f_sr']/1e9:7.2f}"
        except Exception:
            os = f"{'infeasible':^28}"
        try:
            a = best_int(Lt, gamma=GAMMA)
            as_ = f"{round(a['n']):>2} {a['w']*1e6:5.1f} {a['d_out']*1e6:5.0f} {a['Q']:5.1f} {a['f_sr']/1e9:7.2f}"
        except Exception:
            as_ = f"{'INFEASIBLE (SRF<7GHz)':^33}"
        pm = f"{meas.get(Lt, float('nan')):6.2f}" if Lt in meas else "   >10"
        print(f"{Lt:>3} | {os} | {as_} | {pm}")


if __name__ == "__main__":
    main()
