#!/usr/bin/env python3
"""
analyze_lhs.py
==============
Compare every GP-predicted pi-model component against Palace on the Latin-hypercube sweep,
using Niknejad-Meyer's extraction from the raw 2-port Y-matrix (single-ended, the device
treated exactly as in their 1998 paper):

  series branch  Z_se = -1/Y12              ->  L_s = Im(Z_se)/w   (at 0.5 GHz, where the
                                                feedthrough cap C_BR is negligible)
  feedthrough    C_s = C_BR = 1/(w_br^2 L_s) ->  from the series-branch self-resonance w_br
                                                (first Im(Z_se) +->- crossing from low freq)
  shunt branch   Y_sh = Y11 + Y12            ->  C_p = Im(Y_sh)/w,  R_p = 1/Re(Y_sh)
                                                (parallel-RC equiv at 2.5 GHz; Y11+Y12
                                                exactly removes the series branch in a pi)

R_s is deliberately NOT compared: the series loss is a small real part swamped by the large
reactance (off resonance) or by the C_BR cancellation (near resonance), so it is not
cleanly extractable at 2.5 GHz from this data.

Errors are reported as 100*(pred - meas)/meas, so + = the GP model OVER-predicts.

    python3 scripts/analyze_lhs.py
"""
import csv
import math

import numpy as np
import skrf as rf

import fit_cbr as fc                 # REPO, best_s2p
import generate_lhs_sweep as gls     # predict(), _K (k1,k3,k6,k7,k8 at 2.5 GHz)

REPO = fc.REPO
RT2 = math.sqrt(2)
K = gls._K


def measure(md):
    """Niknejad pi-model components from the 2-port Y-matrix. Returns dict or None."""
    p = fc.best_s2p(md)
    if p is None:
        return None
    nw = rf.Network(str(p)); f = nw.f; w = 2 * np.pi * f
    Y = nw.y
    Y12 = 0.5 * (Y[:, 0, 1] + Y[:, 1, 0])                       # reciprocity symmetrise
    Zse = -1.0 / Y12
    Ysh = 0.5 * ((Y[:, 0, 0] + Y[:, 0, 1]) + (Y[:, 1, 1] + Y[:, 1, 0]))   # avg both ports
    i05 = int(np.argmin(np.abs(f - 0.5e9)))
    i25 = int(np.argmin(np.abs(f - 2.5e9)))
    if Zse[i05].imag <= 0:                                       # not inductive at 0.5 GHz
        return None
    Ls = Zse[i05].imag / w[i05]
    im = Zse.imag; wbr = None
    for k in range(len(f) - 1):                                 # fundamental branch resonance
        if f[k] > 0.3e9 and im[k] > 0 >= im[k + 1]:
            wbr = w[k] + (w[k + 1] - w[k]) * im[k] / (im[k] - im[k + 1]); break
    Cbr = 1.0 / (wbr**2 * Ls) if wbr else np.nan
    Cp = Ysh[i25].imag / w[i25]
    Rp = 1.0 / Ysh[i25].real
    return dict(Ls=Ls, Cbr=Cbr, Cp=Cp, Rp=Rp, fbr=(wbr / (2 * np.pi) if wbr else np.nan))


def predict(n, w_m, s_m, dout_m):
    p = gls.predict(n, w_m, s_m, dout_m)
    l = 8 * p["d_avg"] * n / (1 + RT2)
    p["R_p"] = K["k6"] / (l * w_m)
    return p


def main():
    man = list(csv.DictReader(open(REPO / "scripts/.out/lhs/manifest.csv")))
    out = []
    for r in man:
        md = REPO / r["model_dir"]
        if not md.is_dir():
            continue
        m = measure(md)
        if m is None:
            continue
        n = int(r["N"]); wm = float(r["w_um"]) * 1e-6
        sm = float(r["s_um"]) * 1e-6; dom = float(r["d_out_um"]) * 1e-6
        p = predict(n, wm, sm, dom)
        out.append(dict(N=n, w=float(r["w_um"]), dout=float(r["d_out_um"]), fbr=m["fbr"],
                        L_m=m["Ls"] * 1e9, L_p=p["L"] * 1e9,
                        Cs_m=m["Cbr"] * 1e15, Cs_p=p["C_s"] * 1e15,
                        Cp_m=m["Cp"] * 1e15, Cp_p=p["C_p"] * 1e15,
                        Rp_m=m["Rp"], Rp_p=p["R_p"]))

    def stat(key):
        mm = np.array([o[f"{key}_m"] for o in out])
        pp = np.array([o[f"{key}_p"] for o in out])
        ok = np.isfinite(mm) & np.isfinite(pp) & (mm != 0)
        e = 100 * (pp[ok] - mm[ok]) / mm[ok]
        return ok.sum(), e.mean(), np.sqrt(np.mean(e**2))

    print(f"=== LHS pi-model component validation ({len(out)} coils) ===")
    print("Niknejad extraction; error = 100*(pred-meas)/meas  (+ = GP over-predicts)\n")
    print(f"  {'component':24} {'N':>3}  {'mean':>7}  {'RMS':>6}")
    for key, name in [("L", "L_s  (Im(-1/Y12)/w @0.5)"), ("Cs", "C_s  (C_BR, branch res)"),
                      ("Cp", "C_p  (Im(Y11+Y12)/w @2.5)"), ("Rp", "R_p  (1/Re(Y11+Y12)@2.5)")]:
        nN, mean, rms = stat(key)
        print(f"  {name:24} {nN:>3}  {mean:>+6.1f}%  {rms:>5.1f}%")

    out.sort(key=lambda o: o["L_m"])
    print(f"\n{'N':>2} {'w':>5} {'dout':>5} {'fbr':>5} | "
          f"{'L_m':>5} {'L_p':>5} | {'Cs_m':>5} {'Cs_p':>5} | {'Cp_m':>5} {'Cp_p':>5} | "
          f"{'Rp_m':>5} {'Rp_p':>5}")
    print("-" * 86)
    for o in out:
        fb = f"{o['fbr']:5.2f}" if np.isfinite(o["fbr"]) else "  -  "
        cs_m = f"{o['Cs_m']:5.0f}" if np.isfinite(o["Cs_m"]) else "  -  "
        print(f"{o['N']:>2} {o['w']:>5.1f} {o['dout']:>5.0f} {fb} | "
              f"{o['L_m']:5.1f} {o['L_p']:5.1f} | {cs_m} {o['Cs_p']:5.0f} | "
              f"{o['Cp_m']:5.1f} {o['Cp_p']:5.1f} | {o['Rp_m']:5.0f} {o['Rp_p']:5.0f}")

    with open(REPO / "scripts/.out/lhs/comparison_lhs.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(out[0].keys()), lineterminator="\n")
        wtr.writeheader(); wtr.writerows(out)
    print(f"\nWrote {REPO/'scripts/.out/lhs/comparison_lhs.csv'}")


if __name__ == "__main__":
    main()
