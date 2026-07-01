#!/usr/bin/env python3
"""
fit_inductance_lhs.py
=====================
Refit the Mohan inductance monomial  L[nH] = beta * d_out^a1 * w^a2 * d_avg^a3 * n^a4 * s^a5
(geometry in um) directly to the Palace L_nom on the decorrelated Latin-hypercube sweep.

The cturn locus could not identify this fit (d_out, d_avg, n, w, s nearly collinear there ->
only a1+a3 was determined). The LHS set samples n/w/s/d_avg independently, so the individual
exponents become identifiable. We report per-coefficient std errors + cond(X^T X), the full
6-coefficient fit AND the d_out-only variant (drop d_avg, which is algebraically d_out minus
the winding width so the two are partially redundant), and benchmark both against the current
ASITIC SG13G2 AC coefficients.

L_nom is the Niknejad series inductance L_s = Im(-1/Y12)/w at 0.5 GHz (== differential L_nom
to ~1%).

    python3 scripts/fit_inductance_lhs.py
"""
import csv
import math

import numpy as np
import skrf as rf

import fit_cbr as fc                 # REPO, best_s2p
import generate_lhs_sweep as gls     # COEFFS (current ASITIC AC fit)

REPO = fc.REPO


# LHS sweeps to pool: the original s in [2,7] plus any wider-s follow-ups (lhs_wides*).
# Same manifest schema -> they combine into one decorrelated fit spanning s in [2,30].
def _sweeps():
    base = REPO / "scripts/.out"
    return sorted(d.name for d in base.glob("lhs*")
                  if "_test" not in d.name and (d / "manifest.csv").exists())


def load(sweeps=None):
    rows = []
    for sw in (sweeps or _sweeps()):
        man = REPO / "scripts/.out" / sw / "manifest.csv"
        if not man.exists():
            continue
        for r in csv.DictReader(open(man)):
            md = REPO / r["model_dir"]
            if not md.is_dir():
                continue
            p = fc.best_s2p(md)
            if p is None:
                continue
            nw = rf.Network(str(p)); f = nw.f; w = 2 * np.pi * f
            Y12 = 0.5 * (nw.y[:, 0, 1] + nw.y[:, 1, 0])
            i = int(np.argmin(np.abs(f - 0.5e9)))
            Ls = (-1.0 / Y12)[i].imag / w[i]
            if Ls <= 0:
                continue
            n = int(r["N"]); wu = float(r["w_um"]); su = float(r["s_um"]); do = float(r["d_out_um"])
            davg = do - n * wu - (n - 1) * su
            rows.append((n, wu, su, do, davg, Ls * 1e9))
    return rows


def fit(rows, cols, names):
    X, y = [], []
    for n, wu, su, do, davg, Ln in rows:
        feat = {"1": 1.0, "do": math.log(do), "w": math.log(wu), "davg": math.log(davg),
                "n": math.log(n), "s": math.log(su)}
        X.append([feat[c] for c in cols]); y.append(math.log(Ln))
    X, y = np.array(X), np.array(y)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ c
    s2 = np.sum(resid**2) / (len(y) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(cov))
    err = 100 * (np.exp(X @ c) - np.exp(y)) / np.exp(y)
    print(f"  cond(X^T X) = {np.linalg.cond(X.T@X):.1e}   "
          f"fit: mean {err.mean():+.1f}%  RMS {np.sqrt(np.mean(err**2)):.1f}%  med|.| {np.median(np.abs(err)):.1f}%")
    for nm, ci, si in zip(names, c, se):
        v = math.exp(ci) if nm == "beta" else ci
        print(f"    {nm:8} = {v:9.5f}  +/- {si:.4f}")
    return c


def bench_current(rows):
    C = gls.COEFFS
    err = []
    for n, wu, su, do, davg, Ln in rows:
        Lp = C.beta * do**C.a1 * wu**C.a2 * davg**C.a3 * n**C.a4 * su**C.a5
        err.append(100 * (Lp - Ln) / Ln)
    err = np.array(err)
    print(f"current ASITIC AC coeffs (beta={C.beta:.4g} a1={C.a1:.3f} a2={C.a2:.3f} "
          f"a3={C.a3:.3f} a4={C.a4:.3f} a5={C.a5:.3f}):")
    print(f"  vs Palace L_nom: mean {err.mean():+.1f}%  RMS {np.sqrt(np.mean(err**2)):.1f}%  "
          f"med|.| {np.median(np.abs(err)):.1f}%")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", default="", help="comma-separated sweep dirs under "
                    "scripts/.out/ to fit (default: auto-pool lhs*). Use the rapidfem "
                    "campaign explicitly, e.g. --sweeps rf_lhs,rf_lhs_wides, so it is "
                    "NOT mixed with the Palace lhs* sweeps.")
    a = ap.parse_args()
    sweeps = [s for s in a.sweeps.split(",") if s] or None
    rows = load(sweeps)
    src = ",".join(sweeps) if sweeps else "lhs*"
    print(f"=== Mohan monomial refit on {len(rows)} coils ({src}) ===\n")
    bench_current(rows)

    # regressor collinearity (the reason the cturn locus failed)
    M = np.array([[math.log(do), math.log(wu), math.log(davg), math.log(n), math.log(su)]
                  for n, wu, su, do, davg, Ln in rows])
    R = np.corrcoef(M.T); lab = ["do", "w", "davg", "n", "s"]
    print("\nregressor corr matrix (do,w,davg,n,s):")
    for nm, row in zip(lab, R):
        print("  " + f"{nm:5}" + " ".join(f"{v:+.2f}" for v in row))

    print("\n--- FULL Mohan form (d_out, w, d_avg, n, s) ---")
    fit(rows, ["1", "do", "w", "davg", "n", "s"],
        ["beta", "a1 d_out", "a2 w", "a3 d_avg", "a4 n", "a5 s"])

    print("\n--- d_out-only (drop d_avg; a3=0) ---")
    fit(rows, ["1", "do", "w", "n", "s"],
        ["beta", "a1 d_out", "a2 w", "a4 n", "a5 s"])


if __name__ == "__main__":
    main()
