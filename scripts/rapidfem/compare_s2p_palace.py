#!/usr/bin/env python3
"""
Compare the PULLED rapidfem Touchstone results against Palace, reading the .s2p
files directly (no solver re-run -- unlike compare_rapidfem_palace.py, which
re-invokes simulate_rapidfem and would OOM locally at the production domain).

For each <sweep>/rapidfem/sweep_L*_rapidfem.s2p we extract L,Q at --freq via the
same single-ended Niknejad 2-port formula used by simulate_rapidfem / comparison.csv:
    Z = sqrt(z0)(I+S)(I-S)^-1 sqrt(z0);  Y = Z^-1
    L = -1/(w*Im Y11),  Q = -Im Y11 / Re Y11
and join against comparison.csv (palace_L_nH, palace_Q) by the L-number.

Usage:  python3 scripts/compare_s2p_palace.py [sweep_dir] [--freq 2.5e9]
"""
import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np


def read_touchstone(path):
    """Return (freqs[Hz], S[nf,2,2]) from a 2-port .s2p (RI/MA/DB, any R)."""
    fmt, z0 = "MA", 50.0
    fs, Ss = [], []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("!"):
            continue
        if s.startswith("#"):
            t = s.upper().split()
            if "RI" in t: fmt = "RI"
            elif "DB" in t: fmt = "DB"
            elif "MA" in t: fmt = "MA"
            if "R" in t:
                try: z0 = float(t[t.index("R") + 1])
                except (ValueError, IndexError): pass
            continue
        v = [float(x) for x in s.split()]
        f, a = v[0], v[1:9]
        S = []
        for i in range(0, 8, 2):
            if fmt == "RI": S.append(complex(a[i], a[i + 1]))
            elif fmt == "DB": S.append(10 ** (a[i] / 20) * np.exp(1j * np.deg2rad(a[i + 1])))
            else: S.append(a[i] * np.exp(1j * np.deg2rad(a[i + 1])))
        fs.append(f)
        Ss.append([[S[0], S[2]], [S[1], S[3]]])  # [[S11,S12],[S21,S22]]
    return np.array(fs), np.array(Ss), z0


def L_Q_at(freqs, S, z0, f_target):
    """DIFFERENTIAL L,Q at f_target -- Zdiff = Z11+Z22-Z12-Z21, L=Im(Zdiff)/w,
    Q=Im/Re(Zdiff). This is what comparison.csv (Palace) uses: verified that
    Palace's own s2p reproduces comparison.csv only differentially, not via the
    single-ended Niknejad -1/(w Im Y11) form. These are differential inductors,
    so the differential metric is the physically relevant one anyway."""
    k = int(np.argmin(np.abs(freqs - f_target)))
    if abs(freqs[k] - f_target) > 0.05 * f_target:
        return None, None, freqs[k]
    w = 2 * np.pi * freqs[k]
    I = np.eye(2)
    Z = np.sqrt(z0) * (I + S[k]) @ np.linalg.inv(I - S[k]) * np.sqrt(z0)
    Zd = Z[0, 0] + Z[1, 1] - Z[0, 1] - Z[1, 0]
    L = Zd.imag / w if Zd.imag else float("nan")
    Q = Zd.imag / Zd.real if Zd.real else float("nan")
    return L * 1e9, Q, freqs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", nargs="?", default="scripts/.out/sweep_2p5ghz")
    ap.add_argument("--freq", type=float, default=2.5e9)
    a = ap.parse_args()
    sweep = Path(a.sweep_dir)

    # Palace reference keyed by L-number (target_nH)
    pal = {}
    for r in csv.DictReader((sweep / "comparison.csv").open()):
        pal[int(float(r["target_nH"]))] = (int(r["N"]), float(r["palace_L_nH"]),
                                           float(r["palace_Q"]))

    rows = []
    for f in glob.glob(str(sweep / "rapidfem" / "sweep_L*_rapidfem.s2p")):
        m = re.search(r"sweep_L(\d+)_", Path(f).name)
        if not m:
            continue
        tgt = int(m.group(1))
        freqs, S, z0 = read_touchstone(f)
        rfL, rfQ, fgot = L_Q_at(freqs, S, z0, a.freq)
        N, pL, pQ = pal.get(tgt, (None, float("nan"), float("nan")))
        rows.append((tgt, N, pL, rfL, pQ, rfQ))

    rows.sort(key=lambda x: x[0])
    print(f"\nrapidfem (pulled .s2p) vs Palace @ {a.freq/1e9:.2f} GHz\n")
    print(f"{'L#':>3} {'N':>2} {'palaceL':>8} {'rfL':>8} {'dL%':>7} "
          f"{'palaceQ':>8} {'rfQ':>8}  note")
    dls = []
    for tgt, N, pL, rfL, pQ, rfQ in rows:
        if rfL is None:
            print(f"L{tgt:<2} {N!s:>2} {pL:>8.3f} {'  --':>8} {'':>7} {pQ:>8.2f} "
                  f"{'--':>8}  no 2.5GHz pt"); continue
        dl = (rfL - pL) / pL * 100 if pL else float("nan")
        note = ""
        if rfL < 0: note = "rf past SRF (L<0)"
        elif pL > 50 or pL < 0: note = "palace past SRF"
        elif abs(dl) <= 6: note = "MATCH <=6%"
        if rfL > 0 and pL > 0 and pL < 50:
            dls.append(dl)
        print(f"L{tgt:<2} {N!s:>2} {pL:>8.3f} {rfL:>8.3f} {dl:>+6.1f}% "
              f"{pQ:>8.2f} {rfQ:>8.2f}  {note}")
    if dls:
        import statistics as st
        print(f"\nclean coils (both L>0, below SRF): n={len(dls)}, "
              f"mean dL={st.mean(dls):+.1f}%, median={st.median(dls):+.1f}%, "
              f"range [{min(dls):+.1f}, {max(dls):+.1f}]%")


if __name__ == "__main__":
    main()
