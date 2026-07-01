#!/usr/bin/env python3
"""
Compare the GP predictions for the check coils (scripts/.out/rf_gp_check) against
their rapidfem BACKSIDE-GROUND 2-port EM results (cleps/run_rapidfem_backside.sh).

The coils were DESIGNED by the rapidfem-fit monomial (capped to the fitted box,
w<=18/s<=7), so this closes the loop: GP geometry -> EM -> does the EM reproduce
the GP's L / Q / SRF?  All three are read off the DIFFERENTIAL driving-point
impedance Zdiff = Z11+Z22-Z12-Z21 -- the exact quantity the GP/notebook derive
their L, Q and SRF from (GeometricProgramming.ipynb 'Q factor'; shunt cap C_p+2C_s).
This is what gp_* is built on, so it is the consistent comparison (single-ended
Y11 forms differ, esp. near SRF):

    L    = Im(Zdiff)/w                  @ --l-freq (0.5 GHz, below resonance)
    Q    = Im(Zdiff)/Re(Zdiff)          @ --q-freq (2.5 GHz, operating)
    SRF  = first Im(Zdiff) zero (+ -> -, scanning up from 0.3 GHz)

NOTE: a systematic L offset is EXPECTED -- the monomial was fit on the ground-FRAME
sims, but these EM runs use the backside-ground fixture (no frame). Trends/shape
across the sweep are the signal; the mean offset is the fixture change.

Usage:  python3 scripts/compare_gp_rapidfem.py [sweep_dir] [--q-freq 2.5e9]
"""
import argparse
import csv
import glob
import math
from pathlib import Path

import numpy as np
import skrf as rf


def extract(s2p, l_freq, q_freq):
    """All three from the DIFFERENTIAL driving-point impedance
    Zdiff = Z11+Z22-Z12-Z21 -- the quantity the GP/notebook derive L, Q and SRF
    from (its shunt cap is C_p+2C_s; see GeometricProgramming.ipynb 'Q factor').
    Using Zdiff (not single-ended Y11) keeps the comparison consistent with gp_*."""
    nw = rf.Network(s2p)
    f, w, Z = nw.f, 2 * np.pi * nw.f, nw.z
    Zd = Z[:, 0, 0] + Z[:, 1, 1] - Z[:, 0, 1] - Z[:, 1, 0]
    il = int(np.argmin(np.abs(f - l_freq)))
    iq = int(np.argmin(np.abs(f - q_freq)))
    L = Zd[il].imag / w[il] * 1e9                           # nH, differential series L
    Q = Zd[iq].imag / Zd[iq].real                           # differential Q
    # SRF: first Im(Zdiff) + -> - crossing above 0.3 GHz (inductive -> capacitive)
    im = Zd.imag
    srf = None
    for k in range(len(f) - 1):
        if f[k] > 0.3e9 and im[k] > 0 >= im[k + 1]:
            srf = (f[k] + (f[k + 1] - f[k]) * im[k] / (im[k] - im[k + 1])) / 1e9
            break
    return L, Q, srf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", nargs="?", default="scripts/.out/rf_gp_check")
    ap.add_argument("--l-freq", type=float, default=0.5e9)
    ap.add_argument("--q-freq", type=float, default=2.5e9)
    a = ap.parse_args()
    sweep = Path(a.sweep_dir)

    rows = []
    for r in csv.DictReader((sweep / "manifest.csv").open()):
        md = Path(r["model_dir"])
        cand = sorted(glob.glob(str(md / "rapidfem" / "*_rapidfem.s2p")))
        if not cand:
            rows.append((r, None)); continue
        try:
            rows.append((r, extract(cand[0], a.l_freq, a.q_freq)))
        except Exception as e:
            print(f"  [skip {r['target_nH']}] unreadable s2p {Path(cand[0]).name}: "
                  f"{type(e).__name__}")
            rows.append((r, None))

    print(f"\nGP (rapidfem-fit monomial) vs rapidfem backside-ground EM"
          f"  [L@{a.l_freq/1e9:.1f}GHz, Q@{a.q_freq/1e9:.1f}GHz]\n")
    hdr = (f"{'coil':>4} {'N':>2} {'gpL':>6} {'emL':>6} {'dL%':>6} | "
           f"{'gpQ':>5} {'emQ':>5} {'dQ%':>6} | {'gpSRF':>6} {'emSRF':>6} {'dSRF%':>6}")
    print(hdr); print("-" * len(hdr))
    dL, dQ, dS = [], [], []
    for r, ex in rows:
        tag = f"L{r['target_nH']}"; N = r["N"]
        gL, gQ, gS = float(r["gp_L_nH"]), float(r["gp_Q"]), float(r["gp_SRF_GHz"])
        if ex is None:
            print(f"{tag:>4} {N:>2} {gL:>6.2f} {'   --':>6}"); continue
        eL, eQ, eS = ex
        pL = (eL - gL) / gL * 100
        pQ = (eQ - gQ) / gQ * 100
        dL.append(pL); dQ.append(pQ)
        sS = f"{eS:>6.2f}" if eS else "  >max"
        pS = f"{(eS-gS)/gS*100:>+6.1f}" if eS else "     -"
        if eS:
            dS.append((eS - gS) / gS * 100)
        print(f"{tag:>4} {N:>2} {gL:>6.2f} {eL:>6.2f} {pL:>+6.1f} | "
              f"{gQ:>5.1f} {eQ:>5.1f} {pQ:>+6.1f} | {gS:>6.2f} {sS} {pS}")
    if dL:
        print(f"\nmean dL {np.mean(dL):+.1f}% (median {np.median(dL):+.1f}%);  "
              f"mean dQ {np.mean(dQ):+.1f}%;  "
              f"mean dSRF {np.mean(dS):+.1f}%" if dS else f"\nmean dL {np.mean(dL):+.1f}%")
        print("(a roughly-constant dL is the frame->backside fixture change, not model error)")


if __name__ == "__main__":
    main()
