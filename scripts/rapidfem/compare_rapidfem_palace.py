#!/usr/bin/env python3
"""
Compare the (air-box-fixed) rapidfem extraction against the Palace reference for
every coil in a sweep dir, at the sweep frequency.

Palace reference = the existing comparison.csv (cols palace_L_nH, palace_Q, built
from each coil's deembedded 2-port s2p via the Niknejad single-ended extraction).
For each row we run scripts/simulate_rapidfem.py on the SAME *_forEM.gds with a
Palace-matched port/ground setup, parse its printed L,Q at the target frequency,
and tabulate dL% = (rapidfem-palace)/palace.

Usage:
    python3 scripts/compare_rapidfem_palace.py scripts/.out/sweep_2p5ghz \
        [--ground frame|strip] [--rows L1,L2,...] [--maxh 90 --metal-maxh 35]

Only L1-L6 (single-point 2.5 GHz, below SRF) are physically meaningful single-
point comparisons; higher-N coils are past SRF (see memory rapidfem-integration).
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # scripts/rapidfem/<file> -> repo root


def parse_L_Q(stdout, freq_ghz):
    """Pull (L_nH, Q) for the target frequency from simulate_rapidfem's table:
        f[GHz]     L[nH]        Q
         2.500     1.375     0.88
    """
    for line in stdout.splitlines():
        m = re.match(r"\s*([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.naf]+)\s*$", line)
        if m and abs(float(m.group(1)) - freq_ghz) < 1e-6:
            try:
                return float(m.group(2)), float(m.group(3))
            except ValueError:
                return float(m.group(2)), float("nan")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--ground", choices=("frame", "strip", "none"), default="frame")
    ap.add_argument("--port", choices=("ground", "diff"), default="ground")
    ap.add_argument("--rows", default="", help="comma list of target tags e.g. L1,L2,L3 (default: all)")
    ap.add_argument("--maxh", type=float, default=90.0)
    ap.add_argument("--metal-maxh", type=float, default=35.0)
    ap.add_argument("--pad", type=float, default=1.15)
    ap.add_argument("--air-thick", type=float, default=60.0)
    ap.add_argument("--timeout", type=int, default=560)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sweep = Path(a.sweep_dir)
    comp = sweep / "comparison.csv"
    rows = list(csv.DictReader(comp.open()))
    want = {s.strip() for s in a.rows.split(",") if s.strip()}

    results = []
    for r in rows:
        target = int(float(r["target_nH"]))
        tag = f"L{target}"
        if want and tag not in want:
            continue
        gds = next(sweep.glob(f"sweep_{tag}_N*_forEM.gds"), None)
        if gds is None:
            print(f"[{tag}] no forEM gds, skip"); continue
        freq_ghz = float(r["f_GHz"])
        cmd = [sys.executable, str(REPO / "scripts/rapidfem/simulate_rapidfem.py"), str(gds),
               "--port", a.port, "--ground", a.ground,
               "--maxh", str(a.maxh), "--metal-maxh", str(a.metal_maxh),
               "--pad", str(a.pad), "--air-thick", str(a.air_thick),
               "--freqs", str(freq_ghz * 1e9),
               "--out", f"/tmp/cmp_{tag}.s2p"]
        print(f"[{tag}] N={r['N']} running...", flush=True)
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            rf_L, rf_Q = parse_L_Q(p.stdout, freq_ghz)
            dofs = next((m.group(1) for m in
                         [re.search(r"(\d+) DOFs", p.stdout)] if m), "?")
            if rf_L is None:
                tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
                print(f"[{tag}] FAILED to parse L; tail={tail}")
        except subprocess.TimeoutExpired:
            rf_L = rf_Q = None; dofs = "TIMEOUT"
            print(f"[{tag}] TIMEOUT")

        pL, pQ = float(r["palace_L_nH"]), float(r["palace_Q"])
        dL = (rf_L - pL) / pL * 100 if rf_L is not None else float("nan")
        results.append(dict(tag=tag, N=int(r["N"]), f_GHz=freq_ghz,
                            palace_L=pL, rapidfem_L=rf_L, dL_pct=dL,
                            palace_Q=pQ, rapidfem_Q=rf_Q, dofs=dofs))
        rfL = f"{rf_L:7.3f}" if rf_L is not None else "   FAIL"
        print(f"[{tag}] palace_L={pL:7.3f}  rapidfem_L={rfL}  dL={dL:+6.1f}%  "
              f"(palace_Q={pQ:.1f} rf_Q={rf_Q})  DOFs={dofs}")

    print("\n=== SUMMARY (L in nH) ===")
    print(f"{'tag':>4} {'N':>2} {'f':>5} {'palace_L':>9} {'rapidfem_L':>10} {'dL%':>7}"
          f" {'palace_Q':>9} {'rf_Q':>7}")
    for x in results:
        rfL = f"{x['rapidfem_L']:10.3f}" if x['rapidfem_L'] is not None else "      FAIL"
        rfQ = f"{x['rapidfem_Q']:7.2f}" if x['rapidfem_Q'] is not None else "   FAIL"
        print(f"{x['tag']:>4} {x['N']:>2} {x['f_GHz']:>5.2f} {x['palace_L']:>9.3f}"
              f" {rfL} {x['dL_pct']:>+7.1f} {x['palace_Q']:>9.2f} {rfQ}")

    out = Path(a.out) if a.out else sweep / "rapidfem_vs_palace.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
