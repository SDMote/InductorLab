#!/usr/bin/env python3
"""
fit_cbr.py
==========
Refit the inter-turn capacitance coefficient gamma directly against Niknejad's bridge
capacitance C_BR, instead of against the one-port SRF.

C_BR is the feedthrough cap across the series branch of the pi model (Fig 6b). It is
extracted from the SERIES-BRANCH self-resonance: the branch Z_s = -1/Y12 behaves as
(R_s + jwL_s) || 1/(jw C_BR) and self-resonates at w_br = 1/sqrt(L_s C_BR), so
    C_BR = 1 / (w_br^2 L_s).
Because C_BR isolates the series capacitance from the substrate shunt C_p (which the SRF
mixes in), fitting gamma to it is cleaner. Our model says

    C_BR == C_s = K3 n w^2  (underpass overlap)  +  gamma * c_phys * n * d_avg / s  (turns)

so subtract the (fixed) overlap term and least-squares the inter-turn slope = gamma.

    python3 scripts/fit_cbr.py                 # lhs + lhs_wides (decorrelated; exponents valid)
    python3 scripts/fit_cbr.py lhs lhs_wides   # explicit
"""
import csv
import math
import sys

import numpy as np
import skrf as rf

import spiral_cturn as sct        # REPO, PDK, _EPS_0, GAMMA

REPO = sct.REPO
_PDK = sct.PDK
# overlap (underpass) cap density and the octagon sidewall-plate constant, from the PDK
K3 = _PDK.eps_r_ox * sct._EPS_0 / _PDK.top_vias[0].thickness
C_PHYS = 8 * (math.sqrt(2) - 1) * sct._EPS_0 * _PDK.eps_r_ox * _PDK.top_metals[0].thickness


def _sweeps():  # all decorrelated geometry sweeps (lhs + any lhs_wides*); combine into one fit
    base = REPO / "scripts/.out"
    return sorted(d.name for d in base.glob("lhs*")
                  if "_test" not in d.name and (d / "manifest.csv").exists())


SWEEPS = None     # auto-discovered in load() when not given explicitly


def manifest_path(sw):
    return REPO / "scripts/.out" / sw / "manifest.csv"


def best_s2p(md):
    cands = list(md.rglob("*.s2p"))
    deemb = [p for p in cands if "deembed" in p.name.lower() and "_dc" not in p.name.lower()]
    pool = deemb or [p for p in cands if "_dc" not in p.name.lower()] or cands
    best, bn = None, -1
    for p in pool:
        try:
            n = rf.Network(str(p)).f.size
        except Exception:
            continue
        if n > bn:
            best, bn = p, n
    return best


def extract_cbr(md):
    """C_BR from the series-branch self-resonance of Z_s = -1/Y12."""
    p = best_s2p(md)
    if p is None:
        return None
    nw = rf.Network(str(p)); f = nw.f; w = 2 * np.pi * f
    zse = -1.0 / nw.y[:, 0, 1]
    i = int(np.argmin(np.abs(f - 0.5e9))); Ls = zse[i].imag / w[i]
    # FUNDAMENTAL series-branch resonance = first Im(Z_s) + -> - crossing scanning up from
    # low freq (the L_s||C_BR parallel resonance; higher re-crossings are spurious distributed
    # modes). The branch resonance drops below 3 GHz for big/high-L coils, so no fixed floor --
    # just skip sub-0.3 GHz noise and require the coil be inductive (Im>0) going in.
    im = zse.imag; wbr = None
    for k in range(len(f) - 1):
        if f[k] > 0.3e9 and im[k] > 0 >= im[k + 1]:
            wbr = w[k] + (w[k + 1] - w[k]) * im[k] / (im[k] - im[k + 1]); break
    if wbr is None or Ls <= 0:
        return (Ls, None, None)
    return (Ls, wbr / (2 * math.pi), 1.0 / (wbr**2 * Ls))


def load(sweeps=None):
    pts = []
    for sw in (sweeps or _sweeps()):
        man = manifest_path(sw)
        if not man.exists():
            continue
        for r in csv.DictReader(open(man)):
            md = REPO / r["model_dir"]
            if not md.is_dir():
                continue
            res = extract_cbr(md)
            if res is None:
                continue
            Ls, fbr, Cbr = res
            n = int(r["N"]); wu = float(r["w_um"]) * 1e-6
            s = float(r["s_um"]) * 1e-6; do = float(r["d_out_um"]) * 1e-6
            pts.append(dict(sw=sw, L=int(r["target_nH"]), n=n, w=wu, s=s, do=do,
                            Ls=Ls, fbr=fbr, Cbr=Cbr))
    return pts


def main():
    sweeps = sys.argv[1:] or SWEEPS
    pts = load(sweeps)
    inband = [p for p in pts if p["fbr"] is not None]
    print(f"{len(pts)} models, {len(inband)} with C_BR resonance inside 0.1-10 GHz\n")

    # feature x = c_phys * n * d_avg / s ; target y = C_BR - overlap
    rows = []
    for p in inband:
        davg = p["do"] - p["n"] * p["w"] - (p["n"] - 1) * p["s"]
        Covl = K3 * p["n"] * p["w"]**2
        x = C_PHYS * p["n"] * davg / p["s"]
        rows.append((p, Covl, x, p["Cbr"] - Covl))
    X = np.array([x for *_, x, _ in [(r[0], r[1], r[2], r[3]) for r in rows]])
    Y = np.array([r[3] for r in rows])

    # gamma through the origin (overlap already removed)
    gamma0 = float(np.sum(X * Y) / np.sum(X * X))
    # gamma + free floor (extra series cap our model misses)
    A = np.column_stack([np.ones_like(X), X])
    (floor, gamma1), *_ = np.linalg.lstsq(A, Y, rcond=None)

    def report(name, gamma, floor):
        err = []
        for p, Covl, x, _ in rows:
            Cmod = Covl + floor + gamma * x
            err.append(100 * (Cmod - p["Cbr"]) / p["Cbr"])
        err = np.array(err)
        print(f"{name}: gamma = {gamma:.3f}  floor = {floor*1e15:+.1f} fF   "
              f"C_BR fit: mean {err.mean():+.1f}%  RMS {np.sqrt(np.mean(err**2)):.1f}%")

    print(f"current model: gamma = {sct.GAMMA:.3f}, floor = 0")
    report("  current   ", sct.GAMMA, 0.0)
    print()
    report("origin fit  ", gamma0, 0.0)
    report("floor fit   ", gamma1, floor)

    # free monomial on the turn residual:  C_BR - overlap = C * n^a * d_avg^b * s^c
    # (still a monomial -> GP-legal). Reveals whether n/d_avg/s scaling, not gamma, is off.
    Xn, Yl = [], []
    for p, Covl, x, _ in rows:
        davg = p["do"] - p["n"] * p["w"] - (p["n"] - 1) * p["s"]
        res = p["Cbr"] - Covl
        if res <= 0:
            continue
        Xn.append([1.0, math.log(p["n"]), math.log(davg), math.log(p["s"])])
        Yl.append(math.log(res))
    Xn, Yl = np.array(Xn), np.array(Yl)
    coef, *_ = np.linalg.lstsq(Xn, Yl, rcond=None)
    em = 100 * (np.exp(Xn @ coef) - np.exp(Yl)) / np.exp(Yl)
    Ceq = math.exp(coef[0])
    print(f"\nfree monomial  C_turn = C * n^{coef[1]:.2f} * d_avg^{coef[2]:.2f} * s^{coef[3]:.2f}"
          f"   (current model: n^1 d_avg^1 s^-1)")
    print(f"  C = {Ceq:.3e}  (gamma-equiv at n^1 davg^1 s^-1 not directly comparable)"
          f"   turn fit: mean {em.mean():+.1f}%  RMS {np.sqrt(np.mean(em**2)):.1f}%")
    print()

    # per-point table with the origin-fit gamma
    print(f"{'sw':>6} {'L':>3} {'N':>2} | {'fbr':>5} {'C_BR':>6} {'overlap':>7} {'turn':>6} "
          f"{'Cmod':>6} {'err%':>6}")
    print("-" * 60)
    for p, Covl, x, _ in rows:
        turn = gamma0 * x
        Cmod = Covl + turn
        print(f"{p['sw']:>6} {p['L']:>3} {p['n']:>2} | {p['fbr']/1e9:5.2f} {p['Cbr']*1e15:6.1f} "
              f"{Covl*1e15:7.1f} {turn*1e15:6.1f} {Cmod*1e15:6.1f} "
              f"{100*(Cmod-p['Cbr'])/p['Cbr']:+6.1f}")


if __name__ == "__main__":
    main()
