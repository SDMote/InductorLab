#!/usr/bin/env python3
"""
analyze_sweep.py
================
Compare the operating-L-targeted GP (Palace coefficients) against full-wave EM across the
1-20 nH design sweep. For each model, extract from Palace (Niknejad pi-model on the 2-port
Y-matrix):
  L_op  = Im(Z_diff)/w at the operating frequency  (what the GP targets)
  L_dc  = Im(Z_diff)/w at 0.5 GHz                  (the DC/geometric value)
  Q_op  = -Im(Y11)/Re(Y11) at the operating freq   (Niknejad eq 16, NOT Im(Z)/Re)
  SRF   = first Im(Y11) - -> + crossing
and compare to the GP's predictions (gp_Lop, gp_Q, gp_SRF in the manifest).

Headline: does an "L nH" design actually read L nH at the operating frequency in Palace?

    python3 scripts/analyze_sweep.py
"""
import csv

import numpy as np
import skrf as rf

import fit_cbr as fc
REPO = fc.REPO


def measure(md, fop):
    p = fc.best_s2p(md)
    if p is None:
        return None
    nw = rf.Network(str(p)); f = nw.f; w = 2 * np.pi * f; Z = nw.z; Y = nw.y
    zd = Z[:, 0, 0] + Z[:, 1, 1] - Z[:, 0, 1] - Z[:, 1, 0]
    Y11 = 0.5 * (Y[:, 0, 0] + Y[:, 1, 1])
    idc = int(np.argmin(np.abs(f - 0.5e9))); iop = int(np.argmin(np.abs(f - fop)))
    Ldc = zd[idc].imag / w[idc] * 1e9
    Lop = zd[iop].imag / w[iop] * 1e9
    Q = -Y11[iop].imag / Y11[iop].real if Y11[iop].real else np.nan
    im = Y11.imag; srf = np.nan
    for k in range(len(f) - 1):
        if im[k] < 0 <= im[k + 1]:
            srf = (f[k] + (f[k + 1] - f[k]) * (-im[k]) / (im[k + 1] - im[k])) / 1e9; break
    return Ldc, Lop, Q, srf


def main():
    man = REPO / "scripts/.out/sweep/manifest.csv"
    if not man.exists():
        print("No sweep manifest yet."); return
    rows = list(csv.DictReader(open(man)))
    print("=== 1-20 nH sweep: operating-targeted GP vs Palace ===")
    print(f"{'tgt':>3} {'f':>4} {'N':>2} | {'gpLop':>5} {'L_op':>5} {'dLop%':>6} | "
          f"{'L_dc':>5} | {'gpQ':>5} {'Q':>5} | {'gpSRF':>6} {'SRF':>5}")
    print("-" * 70)
    dLop, dQ, dSRF = [], [], []
    for r in rows:
        md = REPO / r["model_dir"]
        if not md.is_dir():
            continue
        fop = float(r["op_GHz"]) * 1e9
        m = measure(md, fop)
        if m is None:
            continue
        Ldc, Lop, Q, srf = m
        gpLop = float(r["gp_Lop_nH"]); gpQ = float(r["gp_Q"]); gpSRF = float(r["gp_SRF_GHz"])
        e = 100 * (gpLop - Lop) / Lop
        ss = f"{srf:5.2f}" if np.isfinite(srf) else "  >15"
        print(f"{r['target_nH']:>3} {r['op_GHz']:>4} {r['N']:>2} | {gpLop:5.2f} {Lop:5.2f} {e:+6.1f} | "
              f"{Ldc:5.2f} | {gpQ:5.1f} {Q:5.1f} | {gpSRF:6.2f} {ss}")
        dLop.append(e); dQ.append(100 * (gpQ - Q) / Q if Q else np.nan)
        if np.isfinite(srf):
            dSRF.append(100 * (gpSRF - srf) / srf)

    def stat(name, a):
        a = np.array([x for x in a if np.isfinite(x)])
        print(f"  {name:28}: mean {a.mean():+5.1f}%  RMS {np.sqrt(np.mean(a**2)):4.1f}%")
    print()
    stat("GP L_op vs Palace L_op", dLop)
    stat("GP Q vs Palace Q", dQ)
    stat("GP SRF vs Palace SRF", dSRF)


if __name__ == "__main__":
    main()
